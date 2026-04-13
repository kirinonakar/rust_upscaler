#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
slint::include_modules!();

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView};
use image::codecs::png::{PngEncoder, CompressionType, FilterType};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use glob::glob;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use half::f16;

use std::panic;

// --- Slint HWND and Hooking (Windows only) ---
static APP_WINDOW_HANDLE: OnceLock<slint::Weak<MainWindow>> = OnceLock::new();
#[cfg(target_os = "windows")]
static mut ORIGINAL_WNDPROC: Option<isize> = None;

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::WindowsAndMessaging::{WNDPROC, CallWindowProcW, SetWindowLongPtrW, GWLP_WNDPROC, WM_DROPFILES};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::Shell::{DragQueryFileW, DragFinish, DragAcceptFiles};

#[cfg(target_os = "windows")]
unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_DROPFILES {
        let hdrop = wparam as windows_sys::Win32::UI::Shell::HDROP;
        let mut path_buf = [0u16; 1024];
        let count = DragQueryFileW(hdrop, 0xFFFFFFFF, std::ptr::null_mut(), 0);
        let mut paths = Vec::new();
        for i in 0..count {
            let len = DragQueryFileW(hdrop, i, path_buf.as_mut_ptr(), 1024);
            if len > 0 {
                paths.push(String::from_utf16_lossy(&path_buf[..len as usize]));
            }
        }
        if !paths.is_empty() {
            let paths_str = paths.join("|");
            if let Some(weak) = APP_WINDOW_HANDLE.get() {
                let weak_clone = weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = weak_clone.upgrade() {
                        ui.invoke_files_dropped(slint::SharedString::from(paths_str.as_str()));
                    }
                });
            }
        }
        DragFinish(hdrop);
        return 0;
    }
        if let Some(orig) = ORIGINAL_WNDPROC {
            CallWindowProcW(core::mem::transmute::<isize, WNDPROC>(orig), hwnd, msg, wparam, lparam)
        } else {
            windows_sys::Win32::UI::WindowsAndMessaging::DefWindowProcW(hwnd, msg, wparam, lparam)
        }
}

// --- General Model Trait / Enum ---
enum ModelType {
    Onnx(Session),
}

impl ModelType {
    fn forward(&mut self, x: &Tensor, tile_size: usize) -> Result<Tensor> {
        match self {
            ModelType::Onnx(s) => {
                let device = x.device().clone();
                let (b, c, h, w) = x.dims4().map_err(anyhow::Error::msg)?;
                
                // This is now purely for non-tiled or simple cases if called directly.
                // tiling is handled at a higher level in process_image for better progress reporting.
                if h > tile_size || w > tile_size {
                    // Fallback to tiling without callback if called this way
                    Self::forward_tiled(x, s, tile_size, |_, _| {})
                } else {
                    println!("[ONNX] Standard inference. Shape: [{}, {}, {}, {}]", b, c, h, w);
                    
                    // Ensure dimensions are multiples of 32
                    let h_32 = (h + 31) / 32 * 32;
                    let w_32 = (w + 31) / 32 * 32;
                    
                    let x_padded = if h_32 > h || w_32 > w {
                        x.pad_with_zeros(2, 0, h_32 - h)?
                         .pad_with_zeros(3, 0, w_32 - w)?
                    } else {
                        x.clone()
                    };

                    let data = x_padded.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                    let input_metas: Vec<(String, bool)> = s.inputs().iter().map(|i| {
                        (i.name().to_string(), format!("{:?}", i).contains("Float16"))
                    }).collect();

                    let mut input_feed = Vec::new();
                    for (name, is_fp16) in input_metas {
                        if name == "alpha" {
                            let alpha_val = Value::from_array((Vec::<i64>::new(), vec![1i64])).map_err(anyhow::Error::msg)?;
                            input_feed.push((name, alpha_val.into_dyn()));
                        } else {
                            if is_fp16 {
                                let data_f16: Vec<f16> = data.iter().map(|&f| f16::from_f32(f)).collect();
                                let input_val = Value::from_array(([b, c, h_32, w_32], data_f16)).map_err(anyhow::Error::msg)?;
                                input_feed.push((name, input_val.into_dyn()));
                            } else {
                                let input_val = Value::from_array(([b, c, h_32, w_32], data.clone())).map_err(anyhow::Error::msg)?;
                                input_feed.push((name, input_val.into_dyn()));
                            }
                        }
                    }

                    let start = std::time::Instant::now();
                    let outputs = s.run(input_feed.iter().map(|(k, v)| (k.as_str(), v)).collect::<Vec<_>>()).map_err(anyhow::Error::msg)?;
                    println!("[ONNX] Inference completed in {:?}", start.elapsed());
                    
                    let output_val = outputs.iter().next().map(|(_, v)| v).ok_or_else(|| anyhow::anyhow!("No outputs from model"))?;
                    
                    let (dims, output_vec) = if let Ok((dims, slice)) = output_val.try_extract_tensor::<f32>() {
                        (dims.iter().map(|&d| d as usize).collect::<Vec<_>>(), slice.to_vec())
                    } else if let Ok((dims, slice)) = output_val.try_extract_tensor::<f16>() {
                        (dims.iter().map(|&d| d as usize).collect::<Vec<_>>(), slice.iter().map(|&f| f.to_f32()).collect())
                    } else {
                        return Err(anyhow::anyhow!("Unsupported output type"));
                    };

                    let t_out = if dims.len() == 4 {
                        Tensor::from_vec(output_vec, (dims[0], dims[1], dims[2], dims[3]), &device).map_err(anyhow::Error::msg)?
                    } else if dims.len() == 3 {
                        Tensor::from_vec(output_vec, (1, dims[0], dims[1], dims[2]), &device).map_err(anyhow::Error::msg)?
                    } else {
                        return Err(anyhow::anyhow!("Unsupported output shape: {:?}", dims));
                    };

                    // Crop back to original scale
                    let scale = dims[2] / h_32;
                    if h_32 > h || w_32 > w {
                        Ok(t_out.narrow(2, 0, h * scale)?.narrow(3, 0, w * scale)?)
                    } else {
                        Ok(t_out)
                    }
                }
            }
        }
    }

    fn forward_tiled<F>(x: &Tensor, s: &mut Session, tile_size: usize, mut progress_cb: F) -> Result<Tensor> 
    where F: FnMut(usize, usize) {
        let start_all = std::time::Instant::now();
        let device = x.device().clone();
        let (b, c, h, w) = x.dims4().map_err(anyhow::Error::msg)?;
        let input_metas: Vec<(String, bool)> = s.inputs().iter().map(|i| {
            (i.name().to_string(), format!("{:?}", i).contains("Float16"))
        }).collect();
        
        // Settings for high quality tiling
        let pad = 32; // Overlap/Padding size to avoid edge artifacts. 32 is standard for high quality.
        let mut scale = 0;
        
        let mut row_tensors = Vec::new();
        
        let num_y_steps = (h + tile_size - 1) / tile_size;
        let num_x_steps = (w + tile_size - 1) / tile_size;
        let total_tiles = num_y_steps * num_x_steps;
        let mut tile_counter = 0;

        for y in (0..h).step_by(tile_size) {
            let mut col_tensors = Vec::new();
            for x_pos in (0..w).step_by(tile_size) {
                tile_counter += 1;
                progress_cb(tile_counter, total_tiles);

                let th = (tile_size).min(h - y);
                let tw = (tile_size).min(w - x_pos);
                
                // Calculate padded coordinates
                let y1 = if y > pad { y - pad } else { 0 };
                let y2 = (y + th + pad).min(h);
                let x1 = if x_pos > pad { x_pos - pad } else { 0 };
                let x2 = (x_pos + tw + pad).min(w);
                
                let p_th = y2 - y1;
                let p_tw = x2 - x1;
                
                // Ensure tile dimensions are multiples of 32
                let p_th_32 = (p_th + 31) / 32 * 32;
                let p_tw_32 = (p_tw + 31) / 32 * 32;

                let tile = x.narrow(2, y1, p_th)?.narrow(3, x1, p_tw)?;
                
                // Pad if necessary
                let tile = if p_th_32 > p_th || p_tw_32 > p_tw {
                    tile.pad_with_zeros(2, 0, p_th_32 - p_th)?
                        .pad_with_zeros(3, 0, p_tw_32 - p_tw)?
                } else {
                    tile
                };

                let data = tile.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                
                let mut input_feed = Vec::new();
                for (name, is_fp16) in input_metas.iter() {
                    if name == "alpha" {
                        let alpha_val = Value::from_array((Vec::<i64>::new(), vec![1i64])).map_err(anyhow::Error::msg)?;
                        input_feed.push((name.clone(), alpha_val.into_dyn()));
                    } else {
                        if *is_fp16 {
                            let data_f16: Vec<f16> = data.iter().map(|&f| f16::from_f32(f)).collect();
                            let input_val = Value::from_array(([b, c, p_th_32, p_tw_32], data_f16)).map_err(anyhow::Error::msg)?;
                            input_feed.push((name.clone(), input_val.into_dyn()));
                        } else {
                            let input_val = Value::from_array(([b, c, p_th_32, p_tw_32], data.clone())).map_err(anyhow::Error::msg)?;
                            input_feed.push((name.clone(), input_val.into_dyn()));
                        }
                    }
                }

                let start_tile = std::time::Instant::now();
                let outputs = s.run(input_feed.iter().map(|(k, v)| (k.as_str(), v)).collect::<Vec<_>>()).map_err(anyhow::Error::msg)?;
                let tile_time = start_tile.elapsed();
                
                let output_val = outputs.iter().next().map(|(_, v)| v).ok_or_else(|| anyhow::anyhow!("No outputs from model"))?;
                
                let (dims, output_vec) = if let Ok((dims, slice)) = output_val.try_extract_tensor::<f32>() {
                    (dims.iter().map(|&d| d as usize).collect::<Vec<_>>(), slice.to_vec())
                } else if let Ok((dims, slice)) = output_val.try_extract_tensor::<f16>() {
                    (dims.iter().map(|&d| d as usize).collect::<Vec<_>>(), slice.iter().map(|&f| f.to_f32()).collect())
                } else {
                    return Err(anyhow::anyhow!("Unsupported output type"));
                };
                
                let out_h = dims[2];
                let out_w = dims[3];
                
                if scale == 0 {
                    scale = out_h / p_th;
                    println!("[Tiling] Detected scale factor: {}", scale);
                }
                
                let t_out = Tensor::from_vec(output_vec, (dims[0], dims[1], out_h, out_w), &device).map_err(anyhow::Error::msg)?;
                
                // Crop out the padding from the upscaled tile
                let crop_y = (y - y1) * scale;
                let crop_x = (x_pos - x1) * scale;
                let crop_h = th * scale;
                let crop_w = tw * scale;
                
                let t_cropped = t_out.narrow(2, crop_y, crop_h)?.narrow(3, crop_x, crop_w)?;
                
                println!("[Tiling] Tile at ({}, {}) processed in {:?}. (Padded: {}x{} -> {}x{}, Cropped: {}x{})", x_pos, y, tile_time, p_tw, p_th, out_w, out_h, crop_w, crop_h);
                col_tensors.push(t_cropped);
            }
            let row = Tensor::cat(&col_tensors, 3).map_err(anyhow::Error::msg)?;
            row_tensors.push(row);
        }
        
        let final_tensor = Tensor::cat(&row_tensors, 2).map_err(anyhow::Error::msg)?;
        println!("[Tiling] Total tiling inference and assembly completed in {:?}", start_all.elapsed());
        Ok(final_tensor)
    }
}

fn get_image_files(dir: &Path, paths: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                get_image_files(&path, paths);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()) {
                if ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"].contains(&ext.as_str()) {
                    paths.push(path);
                }
            }
        }
    }
}

// --- Main Application ---

fn main() -> Result<()> {
    panic::set_hook(Box::new(|info| {
        let msg = format!("PANIC: {:?}", info);
        eprintln!("{}", msg);
    }));

    let ui = MainWindow::new()?;
    let ui_weak = ui.as_weak();
    APP_WINDOW_HANDLE.set(ui_weak.clone()).ok();

    // 1. Scan for ONNX models
    let mut model_files = Vec::new();
    if let Ok(entries) = glob("*.onnx") {
        for entry in entries.flatten() {
            model_files.push(entry.to_string_lossy().into_owned());
        }
    }

    if model_files.is_empty() {
        ui.set_models(std::rc::Rc::new(slint::VecModel::from(vec![slint::SharedString::from("No models found")])).into());
    } else {
        model_files.sort();
        let slint_models: Vec<slint::SharedString> = model_files.iter().map(|s| s.into()).collect();
        ui.set_models(std::rc::Rc::new(slint::VecModel::from(slint_models)).into());
        ui.set_selected_model_index(0);
    }

    // 2. Handle files dropped
    ui.on_files_dropped({
        let ui_weak = ui_weak.clone();
        move |path| {
            let ui = ui_weak.unwrap();
            if ui.get_is_processing() { return; }
            let model_path = ui.get_selected_model().to_string();
            let scale_setting = ui.get_selected_scale().to_string();
            let tile_setting = ui.get_selected_tile().to_string();
            let output_folder_setting = ui.get_output_folder().to_string();

            if model_path == "No models found" {
                ui.set_status_text("Please put .onnx models in the folder".into());
                return;
            }

            let mut paths = Vec::new();
            if path.is_empty() {
                if let Some(files) = rfd::FileDialog::new()
                    .add_filter("Image", &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"])
                    .pick_files() {
                    paths.extend(files);
                }
            } else {
                for p_str in path.split('|') {
                    let p = PathBuf::from(p_str);
                    if p.is_dir() {
                        get_image_files(&p, &mut paths);
                    } else if let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()) {
                        if ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"].contains(&ext.as_str()) {
                            paths.push(p);
                        }
                    } else {
                        paths.push(p);
                    }
                }
            }

            if paths.is_empty() { return; }

            ui.set_is_processing(true);
            ui.set_progress(0.0);

            let ui_thread = ui_weak.clone();
            std::thread::spawn(move || {
                println!("[ONNX] Initializing processing thread...");
                let device = Device::Cpu;

                let session_res: Result<Session> = (|| {
                    use ort::execution_providers::DirectMLExecutionProvider;
                    
                    println!("[ONNX] Attempting to build DirectML environment...");
                    let builder = Session::builder().map_err(anyhow::Error::msg)?
                        .with_optimization_level(GraphOptimizationLevel::Level3).map_err(anyhow::Error::msg)?
                        .with_execution_providers([DirectMLExecutionProvider::default()
                            .with_device_id(0)
                            .build()])
                        .map_err(|e| anyhow::anyhow!("DirectML Initialization Failed: {}", e))?;
                    
                    println!("[ONNX] Builder accepted DirectML. Committing file...");
                    let s = builder.commit_from_file(&model_path).map_err(anyhow::Error::msg)?;
                    println!("[ONNX] Session committed. Model loaded.");
                    Ok(s)
                })();
                
                let session = match session_res {
                    Ok(s) => s,
                    Err(e) => {
                        let err_msg = format!("ONNX Load Error: {}", e);
                        update_status(&ui_thread, err_msg, 0.0);
                        finalize(&ui_thread);
                        return;
                    }
                };

                let mut model = ModelType::Onnx(session);
                let total = paths.len();

                for (i, p) in paths.iter().enumerate() {
                    let filename = p.file_name().unwrap_or_default().to_string_lossy();
                    let status = format!("Processing {} ({} / {}) - Inference start...", filename, i + 1, total);
                    update_status(&ui_thread, status.clone(), i as f32 / total as f32);

                    match process_image(p, &mut model, &device, &scale_setting, &tile_setting, &output_folder_setting, &ui_thread, i, total) {
                        Ok(out_p) => {
                            let done_msg = format!("Finished processing {}!", filename);
                            update_status(&ui_thread, done_msg.clone(), (i + 1) as f32 / total as f32);
                            let _ = slint::invoke_from_event_loop({
                                let ui_weak = ui_thread.clone();
                                move || {
                                    if let Some(ui) = ui_weak.upgrade() {
                                        if let Ok(img) = slint::Image::load_from_path(&out_p) {
                                            ui.set_preview_image(img);
                                            ui.set_has_image(true);
                                        }
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            let err_msg = format!("Process Error: {}", e);
                            update_status(&ui_thread, err_msg, i as f32 / total as f32);
                        }
                    }
                }

                update_status(&ui_thread, "Processing completed!".into(), 1.0);
                finalize(&ui_thread);
            });
        }
    });

    // 3. Setup Windows Drag & Drop (Optional/Support)
    #[cfg(target_os = "windows")]
    {
        let ui_handle = ui_weak.clone();
        slint::Timer::single_shot(std::time::Duration::from_millis(500), move || {
            if let Some(ui) = ui_handle.upgrade() {
                use raw_window_handle::{HasWindowHandle, RawWindowHandle};
                if let Ok(handle) = ui.window().window_handle().window_handle() {
                    if let RawWindowHandle::Win32(h) = handle.as_raw() {
                        let hwnd = h.hwnd.get() as HWND;
                        unsafe {
                            // Revoke Slint's default drag drop to use our own hook
                            windows_sys::Win32::System::Ole::RevokeDragDrop(hwnd);
                            DragAcceptFiles(hwnd, 1);
                            let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, wnd_proc as *const () as isize);
                            if prev != 0 {
                                ORIGINAL_WNDPROC = Some(prev);
                            }
                        }
                    }
                }
            }
        });
    }

    ui.on_select_output_folder({
        let ui_weak = ui_weak.clone();
        move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_output_folder(slint::SharedString::from(folder.to_string_lossy().into_owned()));
                }
            }
        }
    });

    ui.run()?;
    Ok(())
}

fn process_image(
    path: &Path, 
    model: &mut ModelType, 
    device: &Device, 
    scale_setting: &str, 
    tile_setting: &str,
    output_folder_setting: &str,
    ui_thread: &slint::Weak<MainWindow>,
    file_idx: usize,
    total_files: usize
) -> Result<PathBuf> {
    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    let start_msg = format!("[Process] Loading image: {:?}", path);
    println!("{}", start_msg);
    let img = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let (w, h) = img.dimensions();
    let img_rgb = img.to_rgb8();
    let data = img_rgb.into_raw();
    
    let norm_msg = "[Process] Normalizing tensor...".to_string();
    println!("{}", norm_msg);
    // Normalize to [0, 1]
    let tensor = Tensor::from_vec(data, (h as usize, w as usize, 3), device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(1.0 / 255.0, 0.0)?
        .unsqueeze(0)?;

    let fw_msg = "[Process] Starting model forward pass...".to_string();
    println!("{}", fw_msg);

    // Parse tile size
    let tile_size = if tile_setting.contains("256") { 256 } else { 512 };

    // Inference (with progress reporting)
    let output = match model {
        ModelType::Onnx(s) => {
            let (_b, _c, h_in, w_in) = tensor.dims4().map_err(anyhow::Error::msg)?;
            if h_in > tile_size || w_in > tile_size {
                println!("[ONNX] Large image detected ({}x{}), using tiling (size {})...", w_in, h_in, tile_size);
                
                let ui_thread_clone = ui_thread.clone();
                let fname_clone = filename.clone();
                let progress_callback = move |tile_idx: usize, total_tiles: usize| {
                    let file_base_progress = file_idx as f32 / total_files as f32;
                    let file_weight = 1.0 / total_files as f32;
                    let tile_progress = tile_idx as f32 / total_tiles as f32;
                    let total_progress = file_base_progress + (tile_progress * file_weight);
                    
                    let status = format!("Processing {} ({} / {}) - Tile {}/{}", fname_clone, file_idx + 1, total_files, tile_idx, total_tiles);
                    update_status(&ui_thread_clone, status, total_progress);
                };

                ModelType::forward_tiled(&tensor, s, tile_size, progress_callback)?
            } else {
                model.forward(&tensor, tile_size)?
            }
        }
    };
    
    let fw_end_msg = "[Process] Forward pass completed.".to_string();
    println!("{}", fw_end_msg);
    
    // Denormalize
    let dnorm_msg = "[Process] Denormalizing output...".to_string();
    println!("{}", dnorm_msg);

    let start_dnorm = std::time::Instant::now();
    // Move to CPU first and then do operations if needed, or keep on GPU as much as possible
    // For large images, doing affine(255.0) on GPU is faster.
    let output = output.squeeze(0)?.clamp(0.0, 1.0)?.affine(255.0, 0.0)?.to_dtype(DType::U8)?;
    let output = output.permute((1, 2, 0))?.to_device(&Device::Cpu)?;
    let (oh, ow, _) = output.dims3()?;
    let output_data = output.flatten_all()?.to_vec1::<u8>()?;
    println!("[Process] Denormalization and CPU sync took {:?}", start_dnorm.elapsed()); // Changed start_tile to start_dnorm

    let save_msg = format!("[Process] Saving output: {}x{}", ow, oh);
    println!("{}", save_msg);

    let start_save = std::time::Instant::now();
    let mut out_img: DynamicImage = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(ow as u32, oh as u32, output_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create output image"))?
    );
    
    // Scaling logic (Final resizing if target is pixels or x2/x3)
    // The model is x4.
    let (target_w, target_h) = match scale_setting {
        "x2" => (w * 2, h * 2),
        "x3" => (w * 3, h * 3),
        "x4" => (ow as u32, oh as u32),
        "1M Pixels" => calculate_size(w, h, 1_000_000),
        "2M Pixels" => calculate_size(w, h, 2_000_000),
        "3M Pixels" => calculate_size(w, h, 3_000_000),
        "4M Pixels" => calculate_size(w, h, 4_000_000),
        "5M Pixels" => calculate_size(w, h, 5_000_000),
        "6M Pixels" => calculate_size(w, h, 6_000_000),
        _ => (ow as u32, oh as u32),
    };

    if target_w != ow as u32 || target_h != oh as u32 {
        out_img = out_img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    }

    let stem = path.file_stem().unwrap().to_string_lossy();
    let out_path = if output_folder_setting.is_empty() {
        path.with_file_name(format!("{}_upscaled.png", stem))
    } else {
        let mut p = PathBuf::from(output_folder_setting);
        p.push(format!("{}_upscaled.png", stem));
        p
    };
    
    // Fast PNG saving: Use 'Fast' compression and 'NoFilter' to significantly reduce CPU time.
    let file = std::fs::File::create(&out_path)?;
    let writer = std::io::BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(
        writer, 
        CompressionType::Fast, 
        FilterType::NoFilter
    );
    out_img.write_with_encoder(encoder)?;
    println!("[Process] Image saving and post-processing took {:?}", start_save.elapsed());
    Ok(out_path)
}

fn calculate_size(w: u32, h: u32, target_pixels: u32) -> (u32, u32) {
    let aspect = w as f64 / h as f64;
    let new_h = (target_pixels as f64 / aspect).sqrt();
    let new_w = new_h * aspect;
    (new_w as u32, new_h as u32)
}

fn update_status(ui_weak: &slint::Weak<MainWindow>, text: String, progress: f32) {
    let _ = slint::invoke_from_event_loop({
        let ui_weak = ui_weak.clone();
        move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_status_text(text.into());
                ui.set_progress(progress);
            }
        }
    });
}

fn finalize(ui_weak: &slint::Weak<MainWindow>) {
    let _ = slint::invoke_from_event_loop({
        let ui_weak = ui_weak.clone();
        move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_is_processing(false);
            }
        }
    });
}




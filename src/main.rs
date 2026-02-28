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
use ort::value::Value;

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
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        match self {
            ModelType::Onnx(s) => {
                let device = x.device().clone();
                let (b, c, h, w) = x.dims4().map_err(anyhow::Error::msg)?;
                
                // Tiling logic for large images
                // Real-ESRGAN x4 usually needs tiles around 256-512 to be safe on mid-range GPUs
                let tile_size = 512;
                
                if h > tile_size || w > tile_size {
                    println!("[ONNX] Large image detected ({}x{}), using tiling (size {})...", w, h, tile_size);
                    Self::forward_tiled(x, s, tile_size)
                } else {
                    println!("[ONNX] Standard inference. Shape: [{}, {}, {}, {}]", b, c, h, w);
                    let data = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                    let input_name = s.inputs().first().map(|i| i.name().to_string()).unwrap_or_else(|| "input".to_string());
                    
                    let input_val = Value::from_array(([b, c, h, w], data)).map_err(anyhow::Error::msg)?;
                    let start = std::time::Instant::now();
                    let outputs = s.run(ort::inputs![input_name.as_str() => input_val]).map_err(anyhow::Error::msg)?;
                    println!("[ONNX] Inference completed in {:?}", start.elapsed());
                    
                    let output_val = outputs.iter().next().map(|(_, v)| v).ok_or_else(|| anyhow::anyhow!("No outputs from model"))?;
                    let (dims_shape, output_slice) = output_val.try_extract_tensor::<f32>().map_err(anyhow::Error::msg)?;
                    let output_vec = output_slice.to_vec();
                    let dims: Vec<usize> = dims_shape.iter().map(|&d| d as usize).collect();
                    
                    if dims.len() == 4 {
                        Tensor::from_vec(output_vec, (dims[0], dims[1], dims[2], dims[3]), &device).map_err(anyhow::Error::msg)
                    } else if dims.len() == 3 {
                        Tensor::from_vec(output_vec, (1, dims[0], dims[1], dims[2]), &device).map_err(anyhow::Error::msg)
                    } else {
                        Err(anyhow::anyhow!("Unsupported output shape: {:?}", dims))
                    }
                }
            }
        }
    }

    fn forward_tiled(x: &Tensor, s: &mut Session, tile_size: usize) -> Result<Tensor> {
        let start_all = std::time::Instant::now();
        let device = x.device().clone();
        let (b, c, h, w) = x.dims4().map_err(anyhow::Error::msg)?;
        let input_name = s.inputs().first().map(|i| i.name().to_string()).unwrap_or_else(|| "input".to_string());
        
        let mut row_tensors = Vec::new();
        for y in (0..h).step_by(tile_size) {
            let mut col_tensors = Vec::new();
            for x_pos in (0..w).step_by(tile_size) {
                let th = (tile_size).min(h - y);
                let tw = (tile_size).min(w - x_pos);
                
                let tile = x.narrow(2, y, th)?.narrow(3, x_pos, tw)?;
                // Critical: Ensure tile is on CPU before sending to ORT if providers expect that
                // ORT with CUDA normally handles this, but let's be explicit.
                let data = tile.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                let input_val = Value::from_array(([b, c, th, tw], data)).map_err(anyhow::Error::msg)?;
                
                let start_tile = std::time::Instant::now();
                let outputs = s.run(ort::inputs![input_name.as_str() => input_val]).map_err(anyhow::Error::msg)?;
                let tile_time = start_tile.elapsed();
                println!("[Tiling] Tile at ({}, {}) processed in {:?}", x_pos, y, tile_time);
                let output_val = outputs.iter().next().map(|(_, v)| v).ok_or_else(|| anyhow::anyhow!("No outputs"))?;
                let (dims_shape, output_slice) = output_val.try_extract_tensor::<f32>().map_err(anyhow::Error::msg)?;
                
                let t_out = Tensor::from_vec(output_slice.to_vec(), (dims_shape[0] as usize, dims_shape[1] as usize, dims_shape[2] as usize, dims_shape[3] as usize), &device).map_err(anyhow::Error::msg)?;
                col_tensors.push(t_out);
            }
            let row = Tensor::cat(&col_tensors, 3).map_err(anyhow::Error::msg)?;
            row_tensors.push(row);
        }
        
        let final_tensor = Tensor::cat(&row_tensors, 2).map_err(anyhow::Error::msg)?;
        println!("[Tiling] Total tiling inference completed in {:?}", start_all.elapsed());
        Ok(final_tensor)
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
        ui.set_selected_model(model_files[0].clone().into());
    }

    // 2. Handle files dropped
    ui.on_files_dropped({
        let ui_weak = ui_weak.clone();
        move |path| {
            let ui = ui_weak.unwrap();
            if ui.get_is_processing() { return; }
            let model_path = ui.get_selected_model().to_string();
            let scale_setting = ui.get_selected_scale().to_string();

            if model_path == "No models found" {
                ui.set_status_text("Please put .onnx models in the folder".into());
                return;
            }

            let mut paths = Vec::new();
            if path.is_empty() {
                if let Some(files) = rfd::FileDialog::new()
                    .add_filter("Image", &["png", "jpg", "jpeg", "bmp", "webp"])
                    .pick_files() {
                    paths.extend(files);
                }
            } else {
                paths.extend(path.split('|').map(PathBuf::from));
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

                    match process_image(p, &mut model, &device, &scale_setting) {
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

    ui.run()?;
    Ok(())
}

fn process_image(path: &Path, model: &mut ModelType, device: &Device, scale_setting: &str) -> Result<PathBuf> {
    let start_msg = format!("[Process] Loading image: {:?}", path);
    println!("{}", start_msg);
    let img = image::open(path)?;
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
    // Inference
    let output = model.forward(&tensor)?;
    
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
        "2M Pixels" => calculate_size(w, h, 2_000_000),
        "3M Pixels" => calculate_size(w, h, 3_000_000),
        "4M Pixels" => calculate_size(w, h, 4_000_000),
        _ => (ow as u32, oh as u32),
    };

    if target_w != ow as u32 || target_h != oh as u32 {
        out_img = out_img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    }

    let stem = path.file_stem().unwrap().to_string_lossy();
    let out_path = path.with_file_name(format!("{}_upscaled.png", stem));
    
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




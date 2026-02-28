#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
slint::include_modules!();

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use glob::glob;

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

// --- ESRGAN / Real-ESRGAN Architecture (Simplified) ---
// This follows the RRDBNet structure commonly found in ESRGAN pth files.

fn leaky_relu(x: &Tensor, alpha: f64) -> Result<Tensor> {
    let relu = x.relu()?;
    Ok((&relu + (x - &relu)?.affine(alpha, 0.0)?)?)
}

struct ResidualDenseBlock {
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    c4: Conv2d,
    c5: Conv2d,
}

impl ResidualDenseBlock {
    fn new(vb: VarBuilder, n_feat: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let c1 = candle_nn::conv2d(n_feat, 32, 3, cfg, vb.pp("conv1"))?;
        let c2 = candle_nn::conv2d(n_feat + 32, 32, 3, cfg, vb.pp("conv2"))?;
        let c3 = candle_nn::conv2d(n_feat + 64, 32, 3, cfg, vb.pp("conv3"))?;
        let c4 = candle_nn::conv2d(n_feat + 96, 32, 3, cfg, vb.pp("conv4"))?;
        let c5 = candle_nn::conv2d(n_feat + 128, n_feat, 3, cfg, vb.pp("conv5"))?;
        Ok(Self { c1, c2, c3, c4, c5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = leaky_relu(&self.c1.forward(x)?, 0.2)?;
        let x1 = Tensor::cat(&[x, &x1], 1)?;
        let x2 = leaky_relu(&self.c2.forward(&x1)?, 0.2)?;
        let x2 = Tensor::cat(&[x, &x1, &x2], 1)?;
        let x3 = leaky_relu(&self.c3.forward(&x2)?, 0.2)?;
        let x3 = Tensor::cat(&[x, &x1, &x2, &x3], 1)?;
        let x4 = leaky_relu(&self.c4.forward(&x3)?, 0.2)?;
        let x4 = Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1)?;
        let x5 = self.c5.forward(&x4)?;
        Ok((x5.affine(0.2, 0.0)? + x)?)
    }
}

struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    fn new(vb: VarBuilder, n_feat: usize) -> Result<Self> {
        Ok(Self {
            rdb1: ResidualDenseBlock::new(vb.pp("RDB1"), n_feat)?,
            rdb2: ResidualDenseBlock::new(vb.pp("RDB2"), n_feat)?,
            rdb3: ResidualDenseBlock::new(vb.pp("RDB3"), n_feat)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.rdb1.forward(x)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;
        Ok((out.affine(0.2, 0.0)? + x)?)
    }
}

struct RRDBNet {
    conv_first: Conv2d,
    body: Vec<RRDB>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
}

impl RRDBNet {
    fn new(vb: VarBuilder, n_block: usize, n_feat: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv_first = candle_nn::conv2d(3, n_feat, 3, cfg, vb.pp("conv_first"))?;
        let mut body = Vec::new();
        let body_vb = vb.pp("RRDB_trunk");
        for i in 0..n_block {
            body.push(RRDB::new(body_vb.pp(&i.to_string()), n_feat)?);
        }
        let conv_body = candle_nn::conv2d(n_feat, n_feat, 3, cfg, vb.pp("conv_body"))?;
        let conv_up1 = candle_nn::conv2d(n_feat, n_feat, 3, cfg, vb.pp("conv_up1"))?;
        let conv_up2 = candle_nn::conv2d(n_feat, n_feat, 3, cfg, vb.pp("conv_up2"))?;
        let conv_hr = candle_nn::conv2d(n_feat, n_feat, 3, cfg, vb.pp("conv_hr"))?;
        let conv_last = candle_nn::conv2d(n_feat, 3, 3, cfg, vb.pp("conv_last"))?;
        Ok(Self { conv_first, body, conv_body, conv_up1, conv_up2, conv_hr, conv_last })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fea = self.conv_first.forward(x)?;
        let mut trunk = fea.clone();
        for block in &self.body {
            trunk = block.forward(&trunk)?;
        }
        trunk = self.conv_body.forward(&trunk)?;
        let fea = (&fea + &trunk)?;

        let fea = leaky_relu(&self.conv_up1.forward(&(upsample_nearest(&fea, 2)?))?, 0.2)?;
        let fea = leaky_relu(&self.conv_up2.forward(&(upsample_nearest(&fea, 2)?))?, 0.2)?;
        let fea = leaky_relu(&self.conv_hr.forward(&fea)?, 0.2)?;
        let out = self.conv_last.forward(&fea)?;
        Ok(out)
    }
}

fn upsample_nearest(x: &Tensor, scale: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x = x.unsqueeze(4)?.repeat(&[1, 1, 1, 1, scale])?;
    let x = x.reshape((b, c, h, w * scale))?;
    let x = x.unsqueeze(3)?.repeat(&[1, 1, 1, scale, 1])?;
    let x = x.reshape((b, c, h * scale, w * scale))?;
    Ok(x)
}

// --- Main Application ---

fn main() -> Result<()> {
    let ui = MainWindow::new()?;
    let ui_weak = ui.as_weak();
    APP_WINDOW_HANDLE.set(ui_weak.clone()).ok();

    // 1. Scan for models
    let mut model_files = Vec::new();
    if let Ok(entries) = glob("*.pth") {
        for entry in entries.flatten() {
            model_files.push(entry.to_string_lossy().into_owned());
        }
    }
    if let Ok(entries) = glob("*.safetensors") {
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
                ui.set_status_text("Please put .pth or .safetensors in the folder".into());
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
                let device = match Device::new_cuda(0) {
                    Ok(d) => d,
                    Err(_) => Device::Cpu,
                };

                // Load model weights
                let vb = if model_path.ends_with(".safetensors") {
                    unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device) }
                } else {
                    // Try candle's pth loader (only works if state_dict format)
                    VarBuilder::from_pth(&model_path, DType::F32, &device)
                };

                let vb = match vb {
                    Ok(v) => v,
                    Err(e) => {
                        update_status(&ui_thread, format!("Model Error: {}", e), 0.0);
                        finalize(&ui_thread);
                        return;
                    }
                };

                // Initialize RRDBNet (Standard Real-ESRGAN params)
                let model = match RRDBNet::new(vb, 23, 64) {
                    Ok(m) => m,
                    Err(e) => {
                         update_status(&ui_thread, format!("Arch Error: {}", e), 0.0);
                         finalize(&ui_thread);
                         return;
                    }
                };

                let total = paths.len();
                for (i, p) in paths.iter().enumerate() {
                    let filename = p.file_name().unwrap_or_default().to_string_lossy();
                    update_status(&ui_thread, format!("Processing {} ({} / {})", filename, i + 1, total), i as f32 / total as f32);

                    match process_image(p, &model, &device, &scale_setting) {
                        Ok(out_p) => {
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
                            update_status(&ui_thread, format!("Process Error: {}", e), i as f32 / total as f32);
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

fn process_image(path: &Path, model: &RRDBNet, device: &Device, scale_setting: &str) -> Result<PathBuf> {
    let img = image::open(path)?;
    let (w, h) = img.dimensions();
    let img_rgb = img.to_rgb8();
    let data = img_rgb.into_raw();
    
    // Normalize to [0, 1]
    let tensor = Tensor::from_vec(data, (h as usize, w as usize, 3), device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(1.0 / 255.0, 0.0)?
        .unsqueeze(0)?;

    // Inference
    let output = model.forward(&tensor)?;
    
    // Denormalize
    let output = output.squeeze(0)?.clamp(0.0, 1.0)?.affine(255.0, 0.0)?.to_dtype(DType::U8)?;
    let output = output.permute((1, 2, 0))?.to_device(&Device::Cpu)?;
    let (oh, ow, _) = output.dims3()?;
    let out_data = output.flatten_all()?.to_vec1::<u8>()?;

    let mut out_img: DynamicImage = DynamicImage::ImageRgb8(
        ImageBuffer::<Rgb<u8>, _>::from_raw(ow as u32, oh as u32, out_data).unwrap()
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
    out_img.save(&out_path)?;
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

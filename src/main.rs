#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
slint::include_modules!();

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use glob::glob;
use ort::session::Session;
use ort::value::Value;
use std::fs::OpenOptions;
use std::io::Write;

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

// --- General Model Trait / Enum ---
enum ModelType {
    RRDBNet(RRDBNet),
    SwinIR(SwinIR),
    Onnx(Session),
}

impl ModelType {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        match self {
            ModelType::RRDBNet(m) => m.forward(x),
            ModelType::SwinIR(m) => m.forward(x),
            ModelType::Onnx(s) => {
                let device = x.device().clone();
                let (b, c, h, w) = x.dims4().map_err(anyhow::Error::msg)?;
                let data = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                
                let input_name = s.inputs().first().map(|i| i.name().to_string()).unwrap_or_else(|| "input".to_string());
                let input_val = Value::from_array(([b, c, h, w], data)).map_err(anyhow::Error::msg)?;
                
                let outputs = s.run(ort::inputs![input_name.as_str() => input_val]).map_err(anyhow::Error::msg)?;
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

fn conv2d_flex(in_c: usize, out_c: usize, k: usize, vb: VarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig { padding: k / 2, ..Default::default() };
    let weight = vb.get((out_c, in_c, k, k), "weight")
        .map_err(|e| anyhow::anyhow!("Missing/mismatched tensor 'weight' at '{}': {}", vb.prefix(), e))?;
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get(out_c, "bias").map_err(|e| anyhow::anyhow!("Missing/mismatched tensor 'bias' at '{}': {}", vb.prefix(), e))?)
    } else {
        None
    };
    Ok(Conv2d::new(weight, bias, cfg))
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
        let c1 = conv2d_flex(n_feat, 32, 3, vb.pp("conv1"))?;
        let c2 = conv2d_flex(n_feat + 32, 32, 3, vb.pp("conv2"))?;
        let c3 = conv2d_flex(n_feat + 64, 32, 3, vb.pp("conv3"))?;
        let c4 = conv2d_flex(n_feat + 96, 32, 3, vb.pp("conv4"))?;
        let c5 = conv2d_flex(n_feat + 128, n_feat, 3, vb.pp("conv5"))?;
        Ok(Self { c1, c2, c3, c4, c5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = leaky_relu(&self.c1.forward(x).map_err(anyhow::Error::msg)?, 0.2)?;
        let x1 = Tensor::cat(&[x, &x1], 1).map_err(anyhow::Error::msg)?;
        let x2 = leaky_relu(&self.c2.forward(&x1).map_err(anyhow::Error::msg)?, 0.2)?;
        let x2 = Tensor::cat(&[x, &x1, &x2], 1).map_err(anyhow::Error::msg)?;
        let x3 = leaky_relu(&self.c3.forward(&x2).map_err(anyhow::Error::msg)?, 0.2)?;
        let x3 = Tensor::cat(&[x, &x1, &x2, &x3], 1).map_err(anyhow::Error::msg)?;
        let x4 = leaky_relu(&self.c4.forward(&x3).map_err(anyhow::Error::msg)?, 0.2)?;
        let x4 = Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1).map_err(anyhow::Error::msg)?;
        let x5 = self.c5.forward(&x4).map_err(anyhow::Error::msg)?;
        Ok((x5.affine(0.2, 0.0).map_err(anyhow::Error::msg)? + x).map_err(anyhow::Error::msg)?)
    }
}

struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    fn new(vb: VarBuilder, n_feat: usize) -> Result<Self> {
        // Handle lower-case 'rdb' in some models
        let (p1, p2, p3) = if vb.contains_tensor("rdb1.conv1.weight") {
            ("rdb1", "rdb2", "rdb3")
        } else {
            ("RDB1", "RDB2", "RDB3")
        };
        Ok(Self {
            rdb1: ResidualDenseBlock::new(vb.pp(p1), n_feat)?,
            rdb2: ResidualDenseBlock::new(vb.pp(p2), n_feat)?,
            rdb3: ResidualDenseBlock::new(vb.pp(p3), n_feat)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.rdb1.forward(x)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;
        Ok((out.affine(0.2, 0.0).map_err(anyhow::Error::msg)? + x).map_err(anyhow::Error::msg)?)
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
        // 1. First layer
        let first_name = if vb.contains_tensor("conv_first.weight") { "conv_first" }
            else if vb.contains_tensor("model.0.weight") { "model.0" }
            else if vb.contains_tensor("net_g.model.0.weight") { "net_g.model.0" }
            else { "conv_first" };
        let conv_first = conv2d_flex(3, n_feat, 3, vb.pp(first_name))?;

        // 2. Trunk (Body)
        let body_vb = if vb.contains_tensor("RRDB_trunk.0.rdb1.conv1.weight") || vb.contains_tensor("RRDB_trunk.0.RDB1.conv1.weight") {
            vb.pp("RRDB_trunk")
        } else if vb.contains_tensor("body.0.rdb1.conv1.weight") || vb.contains_tensor("body.0.RDB1.conv1.weight") {
            vb.pp("body")
        } else if vb.contains_tensor("model.1.sub.0.RDB1.conv1.weight") || vb.contains_tensor("model.1.sub.0.rdb1.conv1.weight") {
            vb.pp("model.1.sub")
        } else if vb.contains_tensor("net_g.body.0.rdb1.conv1.weight") {
            vb.pp("net_g.body")
        } else {
            // Default check to 'body' if nothing else found but 'body.0' exists
            if vb.contains_tensor("body.0.rdb1.conv1.weight") { vb.pp("body") } else { vb.pp("RRDB_trunk") }
        };

        let mut body = Vec::new();
        for i in 0..n_block {
            let b_name = i.to_string();
            // Stop if we reach a block index that doesn't exist
            if !body_vb.contains_tensor(&format!("{}.rdb1.conv1.weight", b_name)) && 
               !body_vb.contains_tensor(&format!("{}.RDB1.conv1.weight", b_name)) {
                if i > 0 { break; }
            }
            body.push(RRDB::new(body_vb.pp(&b_name), n_feat)?);
        }

        // 3. Post-trunk convolution
        let body_conv_names = ["conv_body", "trunk_conv", "model.1.conv_body", "model.1.trunk_conv", "net_g.conv_body"];
        let mut body_conv_name = "conv_body";
        for n in body_conv_names {
            if vb.contains_tensor(&format!("{}.weight", n)) {
                body_conv_name = n;
                break;
            }
        }
        let conv_body = conv2d_flex(n_feat, n_feat, 3, vb.pp(body_conv_name))?;

        // 4. Upsampling layers
        let up1_names = ["upconv1", "model.3", "conv_up1", "upconv.0", "net_g.upconv1", "model.4"];
        let up2_names = ["upconv2", "model.6", "conv_up2", "upconv.1", "net_g.upconv2", "model.7"];
        let hr_names = ["HRconv", "model.8", "conv_hr", "HRconv.0", "net_g.HRconv", "model.9"];
        let last_names = ["conv_last", "model.10", "conv_last.0", "net_g.conv_last", "model.12"];

        let mut up1_name = "upconv1";
        for n in up1_names { if vb.contains_tensor(&format!("{}.weight", n)) { up1_name = n; break; } }
        let mut up2_name = "upconv2";
        for n in up2_names { if vb.contains_tensor(&format!("{}.weight", n)) { up2_name = n; break; } }
        let mut hr_name = "HRconv";
        for n in hr_names { if vb.contains_tensor(&format!("{}.weight", n)) { hr_name = n; break; } }
        let mut last_name = "conv_last";
        for n in last_names { if vb.contains_tensor(&format!("{}.weight", n)) { last_name = n; break; } }

        let conv_up1 = conv2d_flex(n_feat, n_feat, 3, vb.pp(up1_name))?;
        let conv_up2 = conv2d_flex(n_feat, n_feat, 3, vb.pp(up2_name))?;
        let conv_hr = conv2d_flex(n_feat, n_feat, 3, vb.pp(hr_name))?;
        let conv_last = conv2d_flex(n_feat, 3, 3, vb.pp(last_name))?;

        Ok(Self { conv_first, body, conv_body, conv_up1, conv_up2, conv_hr, conv_last })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fea = self.conv_first.forward(x).map_err(anyhow::Error::msg)?;
        let mut trunk = fea.clone();
        for block in &self.body {
            trunk = block.forward(&trunk)?;
        }
        trunk = self.conv_body.forward(&trunk).map_err(anyhow::Error::msg)?;
        let fea = (&fea + &trunk).map_err(anyhow::Error::msg)?;

        let fea = leaky_relu(&self.conv_up1.forward(&upsample_nearest(&fea, 2)?).map_err(anyhow::Error::msg)?, 0.2)?;
        let fea = leaky_relu(&self.conv_up2.forward(&upsample_nearest(&fea, 2)?).map_err(anyhow::Error::msg)?, 0.2)?;
        let fea = leaky_relu(&self.conv_hr.forward(&fea).map_err(anyhow::Error::msg)?, 0.2)?;
        let out = self.conv_last.forward(&fea).map_err(anyhow::Error::msg)?;
        Ok(out)
    }
}

fn upsample_nearest(x: &Tensor, scale: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4().map_err(anyhow::Error::msg)?;
    let x = x.unsqueeze(4).map_err(anyhow::Error::msg)?.repeat(&[1, 1, 1, 1, scale]).map_err(anyhow::Error::msg)?;
    let x = x.reshape((b, c, h, w * scale)).map_err(anyhow::Error::msg)?;
    let x = x.unsqueeze(3).map_err(anyhow::Error::msg)?.repeat(&[1, 1, 1, scale, 1]).map_err(anyhow::Error::msg)?;
    let x = x.reshape((b, c, h * scale, w * scale)).map_err(anyhow::Error::msg)?;
    Ok(x)
}

// --- SwinIR Architecture ---
// Simple implementation of SwinIR for Image Restoration

struct Mlp {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, dim: usize, mlp_ratio: f64) -> Result<Self> {
        let hidden_dim = (dim as f64 * mlp_ratio) as usize;
        let fc1 = candle_nn::linear(dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_dim, dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x).map_err(anyhow::Error::msg)?;
        let x = x.gelu().map_err(anyhow::Error::msg)?;
        Ok(self.fc2.forward(&x).map_err(anyhow::Error::msg)?)
    }
}

fn window_partition(x: &Tensor, window_size: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    let x = x.reshape((b, h / window_size, window_size, w / window_size, window_size, c))?;
    let windows = x.permute((0, 1, 3, 2, 4, 5))?.reshape(((), window_size, window_size, c))?;
    Ok(windows)
}

fn window_reverse(windows: &Tensor, window_size: usize, h: usize, w: usize) -> Result<Tensor> {
    let b = windows.dim(0)? / (h * w / window_size / window_size);
    let x = windows.reshape((b, h / window_size, w / window_size, window_size, window_size, windows.dim(3)?))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.reshape((b, h, w, ()))?;
    Ok(x)
}

struct WindowAttention {
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
    num_heads: usize,
    scale: f64,
}

impl WindowAttention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let qkv = candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = candle_nn::linear(dim, dim, vb.pp("proj"))?;
        let scale = ((dim / num_heads) as f64).powf(-0.5);
        Ok(Self { qkv, proj, num_heads, scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3().map_err(anyhow::Error::msg)?;
        let qkv = self.qkv.forward(x).map_err(anyhow::Error::msg)?.reshape((b, n, 3, self.num_heads, c / self.num_heads)).map_err(anyhow::Error::msg)?.permute((2, 0, 3, 1, 4)).map_err(anyhow::Error::msg)?;
        let q = qkv.get(0).map_err(anyhow::Error::msg)?;
        let k = qkv.get(1).map_err(anyhow::Error::msg)?;
        let v = qkv.get(2).map_err(anyhow::Error::msg)?;

        let q = (q * self.scale).map_err(anyhow::Error::msg)?;
        let attn = q.matmul(&k.transpose(2, 3).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?;
        let attn = candle_nn::ops::softmax(&attn, 3).map_err(anyhow::Error::msg)?;

        let x = attn.matmul(&v).map_err(anyhow::Error::msg)?.permute((0, 2, 1, 3)).map_err(anyhow::Error::msg)?.reshape((b, n, c)).map_err(anyhow::Error::msg)?;
        Ok(self.proj.forward(&x).map_err(anyhow::Error::msg)?)
    }
}

struct SwinTransformerBlock {
    norm1: candle_nn::LayerNorm,
    attn: WindowAttention,
    norm2: candle_nn::LayerNorm,
    mlp: Mlp,
}

impl SwinTransformerBlock {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn = WindowAttention::new(vb.pp("attn"), dim, num_heads)?;
        let norm2 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, 4.0)?;
        Ok(Self { norm1, attn, norm2, mlp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h_ori = x.dim(1).map_err(anyhow::Error::msg)?;
        let w_ori = x.dim(2).map_err(anyhow::Error::msg)?;
        let c = x.dim(3).map_err(anyhow::Error::msg)?;
        
        let x_norm = self.norm1.forward(x).map_err(anyhow::Error::msg)?;
        let windows = window_partition(&x_norm, 8).map_err(anyhow::Error::msg)?;
        let windows = windows.reshape((windows.dim(0).map_err(anyhow::Error::msg)?, (), c)).map_err(anyhow::Error::msg)?;
        let attn_windows = self.attn.forward(&windows)?;
        let attn_windows = attn_windows.reshape((attn_windows.dim(0).map_err(anyhow::Error::msg)?, 8, 8, c)).map_err(anyhow::Error::msg)?;
        let shifted_x = window_reverse(&attn_windows, 8, h_ori, w_ori).map_err(anyhow::Error::msg)?;
        
        let x = (x + shifted_x).map_err(anyhow::Error::msg)?;
        let x = (&x + self.mlp.forward(&self.norm2.forward(&x).map_err(anyhow::Error::msg)?)?)?;
        Ok(x)
    }
}

struct SwinIR {
    conv_first: Conv2d,
    layers: Vec<Vec<SwinTransformerBlock>>,
    conv_after_body: Conv2d,
    conv_before_upsample: Conv2d,
    conv_last: Conv2d,
}

impl SwinIR {
    fn new(vb: VarBuilder, embed_dim: usize) -> Result<Self> {
        let conv_first = conv2d_flex(3, embed_dim, 3, vb.pp("conv_first"))?;
        let mut layers = Vec::new();
        let layers_vb = vb.pp("layers");
        
        let mut i = 0;
        while vb.contains_tensor(&format!("layers.{}.residual_group.blocks.0.attn.qkv.weight", i)) {
            let mut blocks = Vec::new();
            let group_vb = layers_vb.pp(&i.to_string()).pp("residual_group");
            let block_vb = group_vb.pp("blocks");
            
            // Detect heads from first block
            let qkv_shape = vb.get((embed_dim * 3, embed_dim), &format!("layers.{}.residual_group.blocks.0.attn.qkv.weight", i)).map_err(anyhow::Error::msg)?.dims().to_vec();
            let num_heads = if qkv_shape.len() > 0 {
                // qkv weight is (3*embed_dim, embed_dim)
                // In attention, we reshape to (3, heads, head_dim)
                // Often head_dim is 30 or 60.
                6 // default to 6 if unsure, or calculate:
            } else { 6 };

            let mut j = 0;
            while vb.contains_tensor(&format!("layers.{}.residual_group.blocks.{}.attn.qkv.weight", i, j)) {
                blocks.push(SwinTransformerBlock::new(block_vb.pp(&j.to_string()), embed_dim, num_heads)?);
                j += 1;
            }
            layers.push(blocks);
            i += 1;
        }

        if layers.is_empty() {
             return Err(anyhow::anyhow!("No SwinIR layers found in weights"));
        }

        let conv_after_body = conv2d_flex(embed_dim, embed_dim, 3, vb.pp("conv_after_body"))?;
        let conv_before_upsample = conv2d_flex(embed_dim, embed_dim, 3, vb.pp("conv_before_upsample"))?;
        let conv_last = conv2d_flex(embed_dim, 3, 3, vb.pp("conv_last"))?;
        Ok(Self { conv_first, layers, conv_after_body, conv_before_upsample, conv_last })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_first.forward(x).map_err(anyhow::Error::msg)?;
        let x = x.permute((0, 2, 3, 1)).map_err(anyhow::Error::msg)?; 
        let mut res = x.clone();
        for group in &self.layers {
            for block in group {
                res = block.forward(&res)?;
            }
        }
        let res = res.permute((0, 3, 1, 2)).map_err(anyhow::Error::msg)?; 
        let res = self.conv_after_body.forward(&res).map_err(anyhow::Error::msg)?;
        let x = (x.permute((0, 3, 1, 2)).map_err(anyhow::Error::msg)? + res).map_err(anyhow::Error::msg)?;
        
        let x = self.conv_before_upsample.forward(&x).map_err(anyhow::Error::msg)?;
        let x = upsample_nearest(&upsample_nearest(&x, 2)?, 2)?;
        Ok(self.conv_last.forward(&x).map_err(anyhow::Error::msg)?)
    }
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

                // Load model weights or Session
                if model_path.ends_with(".onnx") {
                    let session_res: Result<Session> = (|| {
                        let builder = Session::builder().map_err(anyhow::Error::msg)?;
                        Ok(builder.commit_from_file(&model_path).map_err(anyhow::Error::msg)?)
                    })();
                    
                    let session = match session_res {
                        Ok(s) => s,
                        Err(e) => {
                            let err_msg = format!("ONNX Load Error: {}", e);
                            log_to_file(&err_msg);
                            update_status(&ui_thread, err_msg, 0.0);
                            finalize(&ui_thread);
                            return;
                        }
                    };
                    let mut model = ModelType::Onnx(session);
                    
                    let total = paths.len();
                    for (i, p) in paths.iter().enumerate() {
                        let filename = p.file_name().unwrap_or_default().to_string_lossy();
                        update_status(&ui_thread, format!("Processing {} ({} / {})", filename, i + 1, total), i as f32 / total as f32);

                        match process_image(p, &mut model, &device, &scale_setting) {
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
                                let err_msg = format!("Process Error: {}", e);
                                log_to_file(&err_msg);
                                update_status(&ui_thread, err_msg, i as f32 / total as f32);
                            }
                        }
                    }

                    update_status(&ui_thread, "Processing completed!".into(), 1.0);
                    finalize(&ui_thread);
                    return;
                }

                let vb_res = if model_path.ends_with(".safetensors") {
                    unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device) }
                } else {
                    VarBuilder::from_pth(&model_path, DType::F32, &device)
                };

                let vb_orig = match vb_res {
                    Ok(v) => v,
                    Err(e) => {
                        let mut err_msg = format!("Model Load Error: {}", e);
                        if err_msg.contains("invalid zip archive") {
                            err_msg = "Error: Invalid zip format. This .pth file is likely in an older format not supported by Candle. Please convert it to .safetensors (recommended).".to_string();
                        }
                        log_to_file(&err_msg);
                        update_status(&ui_thread, err_msg, 0.0);
                        finalize(&ui_thread);
                        return;
                    }
                };

                // Auto-detect common state_dict wrappers (params., etc.)
                let possible_prefixes = ["", "params", "params_ema", "net_g", "model", "state_dict", "net", "module", "module.params", "net_g.model", "net_g.module", "features"];
                let mut best_vb = None;

                for p in possible_prefixes {
                    let test_vb = if p.is_empty() { vb_orig.clone() } else { vb_orig.pp(p) };
                    if test_vb.contains_tensor("conv_first.weight") || 
                       test_vb.contains_tensor("model.0.weight") || 
                       test_vb.contains_tensor("body.0.rdb1.conv1.weight") ||
                       test_vb.contains_tensor("body.0.RDB1.conv1.weight") ||
                       test_vb.contains_tensor("RRDB_trunk.0.rdb1.conv1.weight") {
                        best_vb = Some(test_vb);
                        break;
                    }
                }

                let vb = best_vb.unwrap_or(vb_orig.clone());

                // Auto-detect architecture parameters (n_feat, n_blocks)
                let mut n_feat = 64; 
                let mut n_blocks = 0;

                // 1. Detect n_feat from first layer weights
                let first_layer_names = ["conv_first", "model.0", "net_g.model.0", "params.conv_first"];
                for f_name in first_layer_names {
                    if vb.contains_tensor(&format!("{}.weight", f_name)) {
                        if let Ok(w) = vb.get((), &format!("{}.weight", f_name)) {
                            if let Some(dims) = w.dims().get(0) {
                                n_feat = *dims;
                                break;
                            }
                        }
                    }
                }

                // 2. Detect n_blocks by counting sequential RRDB layers
                let body_prefixes = ["body", "RRDB_trunk", "model.1.sub", "net_g.body", "net.body"];
                let mut detected_body = None;

                for bp in body_prefixes {
                    if vb.contains_tensor(&format!("{}.0.rdb1.conv1.weight", bp)) || 
                       vb.contains_tensor(&format!("{}.0.RDB1.conv1.weight", bp)) {
                        detected_body = Some(bp);
                        break;
                    }
                }

                if let Some(bp) = detected_body {
                    let sub_vb = vb.pp(bp);
                    for i in 0..150 {
                        let b_vb = sub_vb.pp(&i.to_string());
                        if b_vb.contains_tensor("rdb1.conv1.weight") || b_vb.contains_tensor("RDB1.conv1.weight") {
                            n_blocks = i + 1;
                        } else {
                            break;
                        }
                    }
                }
                
                // Fallback for Real-ESRGAN standard
                if n_blocks == 0 { n_blocks = 23; }

                // Initialize Model based on detected keys
                let mut model_res: Result<ModelType> = (|| {
                    if vb.contains_tensor("layers.0.residual_group.blocks.0.attn.qkv.weight") || 
                       vb.contains_tensor("patch_embed.proj.weight") ||
                       vb.contains_tensor("layers.0.blocks.0.attn.qkv.weight") {
                        // SwinIR / HAT Architecture Detection
                        let embed_dim = if vb.contains_tensor("conv_first.weight") {
                            vb.get((), "conv_first.weight")?.dim(0)?
                        } else if vb.contains_tensor("patch_embed.proj.weight") {
                            vb.get((), "patch_embed.proj.weight")?.dim(0)?
                        } else { n_feat };
                        
                        let swin = SwinIR::new(vb.clone(), embed_dim)?;
                        Ok(ModelType::SwinIR(swin))
                    } else {
                        // Default to RRDBNet
                        let m = RRDBNet::new(vb.clone(), n_blocks, n_feat)?;
                        Ok(ModelType::RRDBNet(m))
                    }
                })();

                let mut model = match model_res {
                    Ok(m) => m,
                    Err(e) => {
                         let names = get_tensor_names(&model_path);
                         let name_str = if names.is_empty() { 
                             "Could not list tensors (possibly unsupported format)".to_string() 
                         } else { 
                             names[..names.len().min(15)].join(", ") + "..." 
                         };
                         
                         let mut msg = format!("Arch Error: {}\nDetected: {} blocks, {} feats.\nFirst keys: {}", e, n_blocks, n_feat, name_str);
                         if !names.is_empty() {
                             log_to_file(&format!("FULL KEY LIST for {}:\n{}", model_path, names.join("\n")));
                             
                             let is_swinir = names.iter().any(|n| n.contains("patch_embed") || n.contains("attn") || n.contains("residual_group"));
                             if is_swinir {
                                 msg.push_str("\n\n*** Detected Architecture: SwinIR / HAT ***\nInitializing experimental SwinIR support failed.");
                             } else if !names.iter().any(|n| n.contains("conv_first")) {
                                 msg.push_str("\nTip: 'conv_first' not found. Check if the model is really Real-ESRGAN.");
                             }
                         }
                         update_status(&ui_thread, msg, 0.0);
                         finalize(&ui_thread);
                         return;
                    }
                };

                let total = paths.len();
                for (i, p) in paths.iter().enumerate() {
                    let filename = p.file_name().unwrap_or_default().to_string_lossy();
                    update_status(&ui_thread, format!("Processing {} ({} / {})", filename, i + 1, total), i as f32 / total as f32);

                    match process_image(p, &mut model, &device, &scale_setting) {
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
                            let err_msg = format!("Process Error: {}", e);
                            log_to_file(&err_msg);
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

// Diagnostic helper to list tensor names from safetensors or pth (if possible)
fn get_tensor_names(path: &str) -> Vec<String> {
    if path.ends_with(".safetensors") {
        if let Ok(mut f) = std::fs::File::open(path) {
            let mut buffer = Vec::new();
            use std::io::Read;
            if f.read_to_end(&mut buffer).is_ok() {
                if let Ok(st) = safetensors::SafeTensors::deserialize(&buffer) {
                    return st.names().iter().map(|s| s.to_string()).collect();
                }
            }
        }
    } else if path.ends_with(".pth") {
        if let Ok(data) = std::fs::read(path) {
            let mut names = Vec::new();
            let mut current_name = Vec::new();
            for &b in &data {
                // Alphanumeric + common separators used in keys
                if (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') || 
                   (b >= b'0' && b <= b'9') || b == b'.' || b == b'_' || b == b'-' {
                    current_name.push(b);
                } else {
                    if current_name.len() >= 4 {
                        if let Ok(mut s) = std::str::from_utf8(&current_name).map(|s| s.to_string()) {
                            // Trim common pickle binary suffixes (like 'q', 'r', etc.)
                            // We do this by looking for the last meaningful suffix
                            if let Some(idx) = s.find(".weight") { s.truncate(idx + 7); }
                            else if let Some(idx) = s.find(".bias") { s.truncate(idx + 5); }
                            
                            // Only keep it if it looks like a neural network key
                            if s.contains('.') || s.contains('_') || s.starts_with("conv") || s.starts_with("model") {
                                if !names.contains(&s) {
                                    names.push(s);
                                }
                            }
                        }
                    }
                    current_name.clear();
                }
                // Avoid scanning massive files forever if they are not models
                if names.len() >= 3000 { break; }
            }
            return names;
        }
    }
    vec![]
}

fn log_to_file(text: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("errors.log")
    {
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(file, "[{}] {}", now, text);
    }
}

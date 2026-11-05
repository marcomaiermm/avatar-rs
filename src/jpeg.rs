//! MJPEG camera frames, decoded only where they are looked at. While the face is tracked
//! everything downstream (landmark crop, eyes, mouth, optical flow) reads the face and a margin
//! round it; decoding the rest of a 1080p frame was most of the capture thread's ~12 ms.
//! libjpeg-turbo skips the rows above and below (no IDCT, no colour conversion) and crops the
//! columns. Lost face, mesh view or recording: the whole frame, as before.

use mozjpeg_sys as ffi;
use std::mem;

/// RGB of the whole frame (`w`, `h`, pixels); with `region` = [x0, y0, x1, y1] (frame pixels)
/// only that part is decoded and the rest stays black. Damaged data is an error, not a crash.
pub fn decode(data: &[u8], region: Option<[f32; 4]>) -> Result<(u32, u32, Vec<u8>), String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { decode_unchecked(data, region) }))
        .map_err(|e| e.downcast_ref::<String>().cloned().unwrap_or_else(|| "JPEG nicht lesbar".into()))
}

/// Owns the decoder state so it is freed on every exit, including an error unwinding out of
/// libjpeg. The error manager lives in its own box: libjpeg keeps a pointer to it.
struct Decoder {
    cinfo: Box<ffi::jpeg_decompress_struct>,
    _err: Box<ffi::jpeg_error_mgr>,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { ffi::jpeg_destroy_decompress(&mut self.cinfo) };
    }
}

extern "C-unwind" fn fail(cinfo: &mut ffi::jpeg_common_struct) {
    let code = unsafe { cinfo.err.as_ref() }.map_or(0, |e| e.msg_code);
    // Not a panic: no panic hook, no backtrace for a damaged webcam frame.
    std::panic::resume_unwind(Box::new(format!("Webcam-Bild nicht lesbar (libjpeg {code})")));
}

/// Webcams' MJPEG often carries a few stray bytes; libjpeg would print a warning per frame.
extern "C-unwind" fn quiet(_: &mut ffi::jpeg_common_struct, _: std::os::raw::c_int) {}

unsafe fn decode_unchecked(data: &[u8], region: Option<[f32; 4]>) -> (u32, u32, Vec<u8>) {
    unsafe {
        let mut err: Box<ffi::jpeg_error_mgr> = Box::new(mem::zeroed());
        ffi::jpeg_std_error(&mut err);
        err.error_exit = Some(fail);
        err.emit_message = Some(quiet);
        let mut d = Decoder { cinfo: Box::new(mem::zeroed()), _err: err };
        d.cinfo.common.err = &mut *d._err;
        ffi::jpeg_create_decompress(&mut *d.cinfo);
        let c = &mut *d.cinfo;
        ffi::jpeg_mem_src(c, data.as_ptr(), data.len() as _);
        ffi::jpeg_read_header(c, 1);
        c.out_color_space = ffi::J_COLOR_SPACE::JCS_RGB;
        ffi::jpeg_start_decompress(c);
        let (w, h) = (c.output_width, c.output_height);
        let mut rgb = vec![0u8; w as usize * h as usize * 3];
        let [x0, y0, x1, y1] = region.map_or([0, 0, w, h], |[x0, y0, x1, y1]| {
            let clamp = |v: f32, max: u32| (v.max(0.0) as u32).min(max);
            [clamp(x0, w), clamp(y0, h), clamp(x1.ceil(), w), clamp(y1.ceil(), h)]
        });
        if x1 <= x0 || y1 <= y0 {
            return (w, h, rgb);
        }
        // libjpeg widens the crop to whole blocks and says where it really starts.
        let (mut left, mut width) = (x0, x1 - x0);
        if width < w {
            ffi::jpeg_crop_scanline(c, &mut left, &mut width);
        }
        while c.output_scanline < y0 {
            if ffi::jpeg_skip_scanlines(c, y0 - c.output_scanline) == 0 {
                break;
            }
        }
        while c.output_scanline < y1 {
            let y = c.output_scanline as usize;
            let mut row = rgb[(y * w as usize + left as usize) * 3..][..width as usize * 3].as_mut_ptr();
            if ffi::jpeg_read_scanlines(c, &mut row, 1) == 0 {
                break;
            }
        }
        // The rows below are not needed; dropping the decoder abandons them.
        (w, h, rgb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jpeg(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let v = 128.0 + 60.0 * (x * 0.05).sin() * (y * 0.07).cos();
            [v as u8, (255.0 - v) as u8, ((x + y) as u32 % 256) as u8]
        }).collect();
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90).encode(&rgb, w, h, image::ExtendedColorType::Rgb8).unwrap();
        (out, rgb)
    }

    #[test]
    fn region_matches_the_full_decode_inside_and_is_black_outside() {
        let (data, _) = jpeg(320, 240);
        let (w, h, full) = decode(&data, None).unwrap();
        assert_eq!((w, h), (320, 240));
        let (_, _, part) = decode(&data, Some([101.3, 57.0, 203.0, 150.5])).unwrap();
        for y in 0..240usize {
            for x in 0..320usize {
                let i = (y * 320 + x) * 3;
                if (101..203).contains(&x) && (57..151).contains(&y) {
                    assert_eq!(part[i..i + 3], full[i..i + 3], "at {x},{y}");
                } else if !(96..208).contains(&x) {
                    // Outside the block-widened crop columns nothing is written.
                    assert_eq!(part[i..i + 3], [0, 0, 0], "at {x},{y}");
                }
            }
        }
        let (_, _, empty) = decode(&data, Some([500.0, 0.0, 600.0, 10.0])).unwrap();
        assert!(empty.iter().all(|&v| v == 0), "region off the frame decodes nothing");
    }

    #[test]
    fn damaged_frames_are_errors() {
        let (data, _) = jpeg(64, 48);
        let _ = decode(&data[..40], None); // cut off: error or grey rest, but no crash
        assert!(decode(b"not a jpeg at all", None).is_err());
    }
}

use std::result::Result;

use image::{DynamicImage, GenericImageView};

use crate::error::IFXError;
use crate::features::{Extractor, FeatureVector, IFXFeature};

use super::GlobalFeature;

const ARRAY_ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const ARRAY_COSINE: [[f64; 8]; 8] = [
    [
        3.535534e-01,
        3.535534e-01,
        3.535534e-01,
        3.535534e-01,
        3.535534e-01,
        3.535534e-01,
        3.535534e-01,
        3.535534e-01,
    ],
    [
        4.903926e-01,
        4.157348e-01,
        2.777851e-01,
        9.754516e-02,
        -9.754516e-02,
        -2.777851e-01,
        -4.157348e-01,
        -4.903926e-01,
    ],
    [
        4.619398e-01,
        1.913417e-01,
        -1.913417e-01,
        -4.619398e-01,
        -4.619398e-01,
        -1.913417e-01,
        1.913417e-01,
        4.619398e-01,
    ],
    [
        4.157348e-01,
        -9.754516e-02,
        -4.903926e-01,
        -2.777851e-01,
        2.777851e-01,
        4.903926e-01,
        9.754516e-02,
        -4.157348e-01,
    ],
    [
        3.535534e-01,
        -3.535534e-01,
        -3.535534e-01,
        3.535534e-01,
        3.535534e-01,
        -3.535534e-01,
        -3.535534e-01,
        3.535534e-01,
    ],
    [
        2.777851e-01,
        -4.903926e-01,
        9.754516e-02,
        4.157348e-01,
        -4.157348e-01,
        -9.754516e-02,
        4.903926e-01,
        -2.777851e-01,
    ],
    [
        1.913417e-01,
        -4.619398e-01,
        4.619398e-01,
        -1.913417e-01,
        -1.913417e-01,
        4.619398e-01,
        -4.619398e-01,
        1.913417e-01,
    ],
    [
        9.754516e-02,
        -2.777851e-01,
        4.157348e-01,
        -4.903926e-01,
        4.903926e-01,
        -4.157348e-01,
        2.777851e-01,
        -9.754516e-02,
    ],
];

#[derive(Debug)]
pub struct ColorLayout {
    shape: Vec<Vec<i32>>,
    img_y_size: u32,
    img_x_size: u32,
    img: Option<DynamicImage>,
    num_c_coeff: usize,
    num_y_coeff: usize,
    y_coeff: Vec<i32>,
    cb_coeff: Vec<i32>,
    cr_coeff: Vec<i32>,
}

impl ColorLayout {
    pub fn new() -> Self {
        Self {
            shape: vec![vec![0; 64]; 3],
            img_y_size: 0,
            img_x_size: 0,
            img: None,
            num_c_coeff: 6,
            num_y_coeff: 21,
            y_coeff: vec![0; 64],
            cb_coeff: vec![0; 64],
            cr_coeff: vec![0; 64],
        }
    }

    fn fdct(&mut self) {
        for shape in &mut self.shape {
            let mut dct: [f64; 64] = [0.0; 64];

            for i in 0..8 {
                for j in 0..8 {
                    let mut s = 0.0;
                    for k in 0..8 {
                        s += ARRAY_COSINE[j][k] * shape[8 * i + k] as f64;
                    }
                    dct[8 * i + j] = s;
                }
            }

            for j in 0..8 {
                for i in 0..8 {
                    let mut s = 0.0;
                    for k in 0..8 {
                        s += ARRAY_COSINE[i][k] * dct[8 * k + j];
                    }
                    shape[8 * i + j] = (s + 0.499999).floor() as i32;
                }
            }
        }
    }

    fn convert(&mut self) -> Result<(), IFXError> {
        // Check if self.img is None
        let img: &DynamicImage = match &self.img {
            None => return Err(IFXError::NullImage),
            Some(img) => img,
        };

        let mut sum: Vec<Vec<i32>> = vec![vec![0; 64]; 3];
        let mut count: Vec<i32> = vec![0; 64];

        for y in 0..self.img_y_size {
            for x in 0..self.img_x_size {
                let pixel = img.get_pixel(x as u32, y as u32);

                let y_axis = (y as f32 / (self.img_y_size as f32 / 8.0)) as i32;
                let x_axis = (x as f32 / (self.img_x_size as f32 / 8.0)) as i32;

                let a = ((y_axis << 3) + x_axis) as usize;

                let r = pixel[0] as f64;
                let g = pixel[1] as f64;
                let b = pixel[2] as f64;

                // RGB to YCbCr conversion
                let z = (0.299 * r + 0.587 * g + 0.114 * b) / 256.0;
                sum[0][a] += (219.0 * z + 16.5) as i32; // Y
                sum[1][a] += (224.0 * 0.564 * ((b / 256.0) - z) + 128.5) as i32; // Cb
                sum[2][a] += (224.0 * 0.713 * ((r / 256.0) - z) + 128.5) as i32; // Cr

                count[a] += 1;
            }
        }

        for i in 0..8 {
            for j in 0..8 {
                let a = ((i << 3) + j) as usize;
                for k in 0..3 {
                    if count[a] != 0 {
                        self.shape[k][a] = (sum[k][a] / count[a]) as i32;
                    } else {
                        self.shape[k][a] = 0;
                    }
                }
            }
        }

        Ok(())
    }

    fn create(&mut self) {
        self.fdct();

        self.y_coeff[0] = quant_ydc(self.shape[0][0] >> 3) >> 1;
        self.cb_coeff[0] = quant_cdc(self.shape[1][0] >> 3);
        self.cr_coeff[0] = quant_cdc(self.shape[2][0] >> 3);

        // Quantization and zig-zagging
        for i in 1..64 {
            self.y_coeff[i] = quant_ac((self.shape[0][ARRAY_ZIGZAG[i]]) >> 1) >> 3;
            self.cb_coeff[i] = quant_ac(self.shape[1][ARRAY_ZIGZAG[i]]) >> 3;
            self.cr_coeff[i] = quant_ac(self.shape[2][ARRAY_ZIGZAG[i]]) >> 3;
        }
    }
}

impl Extractor for ColorLayout {
    fn extract(&mut self, image: &DynamicImage) -> Result<(), IFXError> {
        // Convert the image to 8-bit RGBA
        let img_rgb: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image.to_rgba8();

        self.img_y_size = img_rgb.height();
        self.img_x_size = img_rgb.width();

        // Update the struct with the new image
        self.img = Some(DynamicImage::ImageRgba8(img_rgb));

        self.convert()?;
        self.create();

        Ok(())
    }
}

impl FeatureVector for ColorLayout {
    fn get_feature_vector(&self) -> Vec<f64> {
        // Create a vector with the correct size to hold the feature vector
        let mut result: Vec<f64> = vec![0.0; (self.num_y_coeff + self.num_c_coeff * 2) as usize];

        // Copy the Y coefficients into the result vector
        for i in 0..self.num_y_coeff as usize {
            result[i] = self.y_coeff[i] as f64;
        }

        // Copy the Cb and Cr coefficients into the result vector
        for i in 0..self.num_c_coeff as usize {
            result[i + self.num_y_coeff] = self.cb_coeff[i] as f64;
            result[i + self.num_y_coeff + self.num_c_coeff] = self.cr_coeff[i] as f64;
        }

        result
    }
}

impl IFXFeature for ColorLayout {
    fn get_feature_name(&self) -> &str {
        "MPEG-7 Color Layout"
    }

    fn get_descriptor_name(&self) -> &str {
        "CLD"
    }
}

impl GlobalFeature for ColorLayout {}

fn quant_ydc(i: i32) -> i32 {
    match i {
        i if i > 192 => 112 + ((i - 192) >> 2),
        i if i > 160 => 96 + ((i - 160) >> 1),
        i if i > 96 => 32 + (i - 96),
        i if i > 64 => 16 + ((i - 64) >> 1),
        _ => i >> 2,
    }
}

fn quant_cdc(i: i32) -> i32 {
    match i {
        i if i > 191 => 63,
        i if i > 160 => 56 + ((i - 160) >> 2),
        i if i > 144 => 48 + ((i - 144) >> 1),
        i if i > 112 => 16 + (i - 112),
        i if i > 96 => 8 + ((i - 96) >> 1),
        i if i > 64 => (i - 64) >> 2,
        _ => 0,
    }
}

fn quant_ac(i: i32) -> i32 {
    let i = i.clamp(-256, 255);
    let abs_i = i.abs();

    let j = match abs_i {
        abs_i if abs_i > 127 => 64 + (abs_i >> 2),
        abs_i if abs_i > 63 => 32 + (abs_i >> 1),
        _ => abs_i,
    };

    if i < 0 {
        -j + 128
    } else {
        j + 128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_layout_extraction() {
        // Load a test image
        let img: DynamicImage = image::open("./test/01.png").expect("Failed to open image");

        // Create a ColorLayout instance
        let mut cl: ColorLayout = ColorLayout::new();

        // Extract features
        cl.extract(&img).expect("Failed to extract features");

        let vector: Vec<f64> = cl.get_feature_vector();

        assert_eq!(
            vector,
            [
                30.0, 31.0, 27.0, 16.0, 21.0, 16.0, 6.0, 16.0, 16.0, 10.0, 16.0, 14.0, 16.0, 14.0,
                16.0, 22.0, 16.0, 16.0, 16.0, 16.0, 19.0, 31.0, 0.0, 3.0, 16.0, 31.0, 16.0, 32.0,
                0.0, 31.0, 16.0, 8.0, 16.0
            ]
        );
    }
}

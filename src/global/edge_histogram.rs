use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};

use crate::{
    error::IFXError,
    features::{Extractor, FeatureVector, IFXFeature},
};

use super::GlobalFeature;

const QUANT_TABLE: [[f32; 8]; 5] = [
    [
        0.010867, 0.057915, 0.099526, 0.144849, 0.195573, 0.260504, 0.358031, 0.530128,
    ],
    [
        0.012266, 0.069934, 0.125879, 0.182307, 0.243396, 0.314563, 0.411728, 0.564319,
    ],
    [
        0.004193, 0.025852, 0.046860, 0.068519, 0.093286, 0.123490, 0.161505, 0.228960,
    ],
    [
        0.004174, 0.025924, 0.046232, 0.067163, 0.089655, 0.115391, 0.151904, 0.217745,
    ],
    [
        0.006778, 0.051667, 0.108650, 0.166257, 0.224226, 0.285691, 0.356375, 0.450972,
    ],
];

const BINS_COUNT: usize = 80;

enum EdgeBlockType {
    NoEdge,
    VerticalEdge,
    HorizontalEdge,
    NonDirectionalEdge,
    Diagonal45degreeEdge,
    Diagonal135degreeEdge,
}

pub struct EdgeHistogram {
    threshold: u32,
    num_block: i32,
    block_size: u32,
    grey_level: Vec<Vec<i32>>,
    img_y_size: u32,
    img_x_size: u32,
    img: Option<DynamicImage>,
    edge_histogram: Vec<f64>,
    bins: Vec<f32>,
}

impl EdgeHistogram {
    pub fn new() -> Self {
        Self {
            threshold: 11,
            num_block: 1100,
            block_size: 0,
            grey_level: Vec::new(),
            img_y_size: 0,
            img_x_size: 0,
            img: None,
            edge_histogram: vec![0f64; BINS_COUNT],
            bins: vec![0f32; BINS_COUNT],
        }
    }
    fn update_block_size(&mut self) {
        let a = ((self.img_x_size * self.img_y_size) as f64 / self.num_block as f64).sqrt() as i32;
        self.block_size = ((a / 2) as f32 * 2 as f32).floor() as u32;
        if self.block_size == 0 {
            self.block_size = 2;
        }
    }
    fn make_grey_level(&mut self) -> Result<(), IFXError> {
        let img = match &self.img {
            Some(img) => img,
            None => return Err(IFXError::NullImage),
        };

        self.grey_level = vec![vec![0i32; self.img_y_size as usize]; self.img_x_size as usize];
        for x in 0..self.img_x_size {
            for y in 0..self.img_y_size {
                let pixel = img.get_pixel(x, y);
                self.grey_level[x as usize][y as usize] = rgba_to_y(&pixel)
            }
        }
        Ok(())
    }

    fn avg_first_block(&mut self, x: u32, y: u32) -> f32 {
        // Check if the grey level is initialized
        if self.grey_level[x as usize][y as usize] == 0 {
            println!("Grey level not initialized.");
            return 0.0; // Return early with 0.0 to indicate failure
        }

        let mut average_brightness: f32 = 0.0;

        // Iterate over the block size
        for m in 0..(self.block_size >> 1) {
            for n in 0..(self.block_size >> 1) {
                average_brightness += self.grey_level[(x + m) as usize][(y + n) as usize] as f32;
            }
        }

        let bs = self.block_size * self.block_size; // Calculate block size squared
        let div: f32 = 4.0 / bs as f32; // Perform the division

        average_brightness * div // Return the average brightness scaled
    }

    fn avg_second_block(&mut self, x: u32, y: u32) -> f32 {
        // Check if the grey level is initialized
        if self.grey_level[x as usize][y as usize] == 0 {
            println!("Grey level not initialized.");
            return 0.0; // Return early with 0.0 to indicate failure
        }

        let mut average_brightness: f32 = 0.0;

        // Iterate over the second block
        for m in (self.block_size >> 1)..self.block_size {
            for n in 0..(self.block_size >> 1) {
                average_brightness += self.grey_level[(x + m) as usize][(y + n) as usize] as f32;
            }
        }

        let bs = self.block_size * self.block_size; // Calculate block size squared
        let div: f32 = 4.0 / bs as f32; // Perform the division

        average_brightness * div // Return the average brightness scaled
    }

    fn avg_third_block(&mut self, x: u32, y: u32) -> f32 {
        // Check if the grey level is initialized
        if self.grey_level[x as usize][y as usize] == 0 {
            println!("Grey level not initialized.");
            return 0.0; // Return early with 0.0 to indicate failure
        }

        let mut average_brightness: f32 = 0.0;

        // Iterate over the third block
        for m in 0..(self.block_size >> 1) {
            for n in (self.block_size >> 1)..self.block_size {
                average_brightness += self.grey_level[(x + m) as usize][(y + n) as usize] as f32;
            }
        }

        let bs = self.block_size * self.block_size; // Calculate block size squared
        let div: f32 = 4.0 / bs as f32; // Perform the division

        average_brightness * div // Return the average brightness scaled
    }

    fn avg_fourth_block(&mut self, x: u32, y: u32) -> f32 {
        let mut average_brightness: f32 = 0.0;

        // Iterate over the fourth block
        for m in (self.block_size >> 1)..self.block_size {
            for n in (self.block_size >> 1)..self.block_size {
                average_brightness += self.grey_level[(x + m) as usize][(y + n) as usize] as f32;
            }
        }

        let bs = self.block_size * self.block_size; // Calculate block size squared
        let div: f32 = 4.0 / bs as f32; // Perform the division

        average_brightness * div // Return the average brightness scaled
    }

    fn detect_edge(&mut self, x: u32, y: u32) -> EdgeBlockType {
        let average_blocks: [f32; 4] = [
            self.avg_first_block(x, y),
            self.avg_second_block(x, y),
            self.avg_third_block(x, y),
            self.avg_fourth_block(x, y),
        ];

        let edge_filter: [[f32; 4]; 5] = [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [2.0_f32.sqrt(), 0.0, 0.0, -2.0_f32.sqrt()],
            [0.0, 2.0_f32.sqrt(), -2.0_f32.sqrt(), 0.0],
            [2.0, -2.0, -2.0, 2.0],
        ];

        let mut strengths: [f32; 5] = [0.0; 5];

        for e in 0..5 {
            for k in 0..4 {
                strengths[e] += average_blocks[k] * edge_filter[e][k];
            }
            strengths[e] = strengths[e].abs();
        }

        let mut max = strengths[0];
        let mut edge: EdgeBlockType = EdgeBlockType::VerticalEdge;

        if strengths[1] > max {
            max = strengths[1];
            edge = EdgeBlockType::HorizontalEdge;
        }
        if strengths[2] > max {
            max = strengths[2];
            edge = EdgeBlockType::Diagonal45degreeEdge;
        }
        if strengths[3] > max {
            max = strengths[3];
            edge = EdgeBlockType::Diagonal135degreeEdge;
        }
        if strengths[4] > max {
            max = strengths[4];
            edge = EdgeBlockType::NonDirectionalEdge;
        }

        if max < self.threshold as f32 {
            edge = EdgeBlockType::NoEdge;
        }

        edge
    }

    fn draw(&mut self) -> Result<(), IFXError> {
        self.bins = vec![0f32; BINS_COUNT];
        self.make_grey_level()?;
        self.update_block_size();

        let mut count_local: [i32; 16] = [0i32; 16];

        let mut y = 0;
        while y <= self.img_y_size - self.block_size as u32 {
            // TODO: remove x loop
            let mut x = 0;
            while x <= self.img_x_size - self.block_size as u32 {
                let sub_index =
                    (((x << 2) / self.img_x_size) + (((y << 2) / self.img_y_size) << 2)) as usize;
                count_local[sub_index] += 1;

                match self.detect_edge(x, y) {
                    EdgeBlockType::NoEdge => (),
                    EdgeBlockType::VerticalEdge => {
                        self.bins[sub_index * 5] += 1f32;
                    }
                    EdgeBlockType::HorizontalEdge => {
                        self.bins[sub_index * 5 + 1] += 1f32;
                    }
                    EdgeBlockType::Diagonal45degreeEdge => {
                        self.bins[sub_index * 5 + 2] += 1f32;
                    }
                    EdgeBlockType::Diagonal135degreeEdge => {
                        self.bins[sub_index * 5 + 3] += 1f32;
                    }
                    EdgeBlockType::NonDirectionalEdge => {
                        self.bins[sub_index * 5 + 4] += 1f32;
                    }
                }
                x += self.block_size as u32;
            }
            y += self.block_size as u32;
        }

        for k in 0..80 as usize {
            self.bins[k] /= count_local[k / 5] as f32;
        }
        Ok(())
    }
    fn update_histogram(&mut self) {
        for i in 0..80 as usize {
            for j in 0..8 as usize {
                self.edge_histogram[i] = j as f64;
                let quant = if j < 7 {
                    (QUANT_TABLE[i % 5][j] + QUANT_TABLE[i % 5][j + 1]) / 2.0
                } else {
                    1.0
                };
                if self.bins[i] as f32 <= quant {
                    break;
                }
            }
        }
    }
}

impl Extractor for EdgeHistogram {
    fn extract(&mut self, image: &DynamicImage) -> Result<(), IFXError> {
        if image.height() <= 150 && image.width() <= 150 {
            return Err(IFXError::SmallImage);
        }
        // Convert the image to 8-bit RGBA
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = image.to_rgba8();

        self.img_y_size = img.height();
        self.img_x_size = img.width();

        // Update the struct with the new image
        self.img = Some(DynamicImage::ImageRgba8(img));

        self.draw()?;
        self.update_histogram();

        Ok(())
    }
}

impl FeatureVector for EdgeHistogram {
    fn get_feature_vector(&self) -> Vec<f64> {
        self.edge_histogram.clone()
    }
}

impl IFXFeature for EdgeHistogram {
    fn get_feature_name(&self) -> &str {
        "MPEG-7 Edge Histogram"
    }

    fn get_descriptor_name(&self) -> &str {
        "EHD"
    }
}

impl GlobalFeature for EdgeHistogram {}

fn rgba_to_y(pixel: &Rgba<u8>) -> i32 {
    let y: f32 =
        (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) / 256.0;
    (219.0 * y + 16.5) as i32
}


#[cfg(test)]
mod tests {

    use super::*;
    use image::DynamicImage;

    // Helper function to initialize the EdgeHistogram and load the image
    fn setup() -> (EdgeHistogram, DynamicImage) {
        let eh = EdgeHistogram::new();
        let img = image::open("./test/01.png").expect("Failed to open the image");
        (eh, img)
    }

    #[test]
    fn test_extract_features() {
        let (mut eh, img) = setup();
        eh.extract(&img).expect("Failed to extract features");

        let vector: Vec<f64> = eh.get_feature_vector();

        assert!(!vector.is_empty(), "Feature vector should not be empty");
        assert_eq!(
            vector,
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0,
                0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn test_edge_histogram_extraction_error() {
        let (mut eh, _) = setup();
        let invalid_img = DynamicImage::new_rgb8(100, 100); // Example of an invalid image

        // Expecting a specific error
        match eh.extract(&invalid_img) {
            Err(IFXError::SmallImage) => (),
            Err(_) => panic!("Unexpected error type"),
            Ok(_) => panic!("Expected an error, but extract succeeded"),
        }
    }
}

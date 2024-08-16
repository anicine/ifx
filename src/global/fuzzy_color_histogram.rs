use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};

use crate::{
    error::IFXError,
    features::{Extractor, FeatureVector, IFXFeature},
};

use super::GlobalFeature;

const SIZE: usize = 5;
const BINS_COUNT: usize = SIZE * SIZE * SIZE;

const COLORS: [Rgb<u8>; BINS_COUNT] = [
    Rgb([25, 25, 25]),
    Rgb([76, 25, 25]),
    Rgb([127, 25, 25]),
    Rgb([178, 25, 25]),
    Rgb([229, 25, 25]),
    Rgb([25, 76, 25]),
    Rgb([76, 76, 25]),
    Rgb([127, 76, 25]),
    Rgb([178, 76, 25]),
    Rgb([229, 76, 25]),
    Rgb([25, 127, 25]),
    Rgb([76, 127, 25]),
    Rgb([127, 127, 25]),
    Rgb([178, 127, 25]),
    Rgb([229, 127, 25]),
    Rgb([25, 178, 25]),
    Rgb([76, 178, 25]),
    Rgb([127, 178, 25]),
    Rgb([178, 178, 25]),
    Rgb([229, 178, 25]),
    Rgb([25, 229, 25]),
    Rgb([76, 229, 25]),
    Rgb([127, 229, 25]),
    Rgb([178, 229, 25]),
    Rgb([229, 229, 25]),
    Rgb([25, 25, 76]),
    Rgb([76, 25, 76]),
    Rgb([127, 25, 76]),
    Rgb([178, 25, 76]),
    Rgb([229, 25, 76]),
    Rgb([25, 76, 76]),
    Rgb([76, 76, 76]),
    Rgb([127, 76, 76]),
    Rgb([178, 76, 76]),
    Rgb([229, 76, 76]),
    Rgb([25, 127, 76]),
    Rgb([76, 127, 76]),
    Rgb([127, 127, 76]),
    Rgb([178, 127, 76]),
    Rgb([229, 127, 76]),
    Rgb([25, 178, 76]),
    Rgb([76, 178, 76]),
    Rgb([127, 178, 76]),
    Rgb([178, 178, 76]),
    Rgb([229, 178, 76]),
    Rgb([25, 229, 76]),
    Rgb([76, 229, 76]),
    Rgb([127, 229, 76]),
    Rgb([178, 229, 76]),
    Rgb([229, 229, 76]),
    Rgb([25, 25, 127]),
    Rgb([76, 25, 127]),
    Rgb([127, 25, 127]),
    Rgb([178, 25, 127]),
    Rgb([229, 25, 127]),
    Rgb([25, 76, 127]),
    Rgb([76, 76, 127]),
    Rgb([127, 76, 127]),
    Rgb([178, 76, 127]),
    Rgb([229, 76, 127]),
    Rgb([25, 127, 127]),
    Rgb([76, 127, 127]),
    Rgb([127, 127, 127]),
    Rgb([178, 127, 127]),
    Rgb([229, 127, 127]),
    Rgb([25, 178, 127]),
    Rgb([76, 178, 127]),
    Rgb([127, 178, 127]),
    Rgb([178, 178, 127]),
    Rgb([229, 178, 127]),
    Rgb([25, 229, 127]),
    Rgb([76, 229, 127]),
    Rgb([127, 229, 127]),
    Rgb([178, 229, 127]),
    Rgb([229, 229, 127]),
    Rgb([25, 25, 178]),
    Rgb([76, 25, 178]),
    Rgb([127, 25, 178]),
    Rgb([178, 25, 178]),
    Rgb([229, 25, 178]),
    Rgb([25, 76, 178]),
    Rgb([76, 76, 178]),
    Rgb([127, 76, 178]),
    Rgb([178, 76, 178]),
    Rgb([229, 76, 178]),
    Rgb([25, 127, 178]),
    Rgb([76, 127, 178]),
    Rgb([127, 127, 178]),
    Rgb([178, 127, 178]),
    Rgb([229, 127, 178]),
    Rgb([25, 178, 178]),
    Rgb([76, 178, 178]),
    Rgb([127, 178, 178]),
    Rgb([178, 178, 178]),
    Rgb([229, 178, 178]),
    Rgb([25, 229, 178]),
    Rgb([76, 229, 178]),
    Rgb([127, 229, 178]),
    Rgb([178, 229, 178]),
    Rgb([229, 229, 178]),
    Rgb([25, 25, 229]),
    Rgb([76, 25, 229]),
    Rgb([127, 25, 229]),
    Rgb([178, 25, 229]),
    Rgb([229, 25, 229]),
    Rgb([25, 76, 229]),
    Rgb([76, 76, 229]),
    Rgb([127, 76, 229]),
    Rgb([178, 76, 229]),
    Rgb([229, 76, 229]),
    Rgb([25, 127, 229]),
    Rgb([76, 127, 229]),
    Rgb([127, 127, 229]),
    Rgb([178, 127, 229]),
    Rgb([229, 127, 229]),
    Rgb([25, 178, 229]),
    Rgb([76, 178, 229]),
    Rgb([127, 178, 229]),
    Rgb([178, 178, 229]),
    Rgb([229, 178, 229]),
    Rgb([25, 229, 229]),
    Rgb([76, 229, 229]),
    Rgb([127, 229, 229]),
    Rgb([178, 229, 229]),
    Rgb([229, 229, 229]),
];

pub struct FCH {
    histogram: Vec<u8>,
    img: Option<DynamicImage>,
}

impl FCH {
    pub fn new() -> Self {
        Self {
            histogram: vec![0u8; BINS_COUNT],
            img: None,
        }
    }
}

impl Extractor for FCH {
    fn extract(&mut self, image: &image::DynamicImage) -> Result<(), IFXError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(IFXError::SmallImage);
        }

        // Convert the image to 8-bit RGB
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = image.to_rgb8();

        // Update the struct with the new image
        self.img = Some(DynamicImage::ImageRgb8(img));

        let img: &DynamicImage = match &self.img {
            Some(img) => img,
            None => return Err(IFXError::NullImage),
        };

        let mut values = vec![0f32; BINS_COUNT];
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                for k in 0..BINS_COUNT {
                    let r = (COLORS[k][0] as isize - pixel[0] as isize) as f32;
                    let g = (COLORS[k][1] as isize - pixel[1] as isize) as f32;
                    let b = (COLORS[k][2] as isize - pixel[2] as isize) as f32;
                    let dist = (r * r) + (g * g) + (b * b);
                    values[k] += 10.0 / (dist + 1.0).sqrt()
                }
            }
        }

        let mut max: f32 = 0f32;
        for v in 0..BINS_COUNT {
            if values[v] > max {
                max = values[v];
            }
        }

        for v in 0..BINS_COUNT {
            self.histogram[v] = (values[v] / max * 255.0) as u8;
        }

        Ok(())
    }
}

impl FeatureVector for FCH {
    fn get_feature_vector(&self) -> Vec<f64> {
        self.histogram.iter().map(|&x| x as f64).collect()
    }
}

impl IFXFeature for FCH {
    fn get_feature_name(&self) -> &str {
        "Fuzzy Color Histogram"
    }

    fn get_descriptor_name(&self) -> &str {
        "FCHD"
    }
}

impl GlobalFeature for FCH {}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    // Helper function to initialize the FCH and load the image
    fn setup() -> (FCH, DynamicImage) {
        let fch = FCH::new();
        let img = image::open("./test/01.png").expect("Failed to open the image");
        (fch, img)
    }

    #[test]
    fn test_extract_features() {
        let (mut fch, img) = setup();

        fch.extract(&img).expect("Failed to extract features");

        let vector: Vec<f64> = fch.get_feature_vector();

        assert!(!vector.is_empty(), "Feature vector should not be empty");
        assert_eq!(
            vector,
            [
                122.0, 129.0, 142.0, 172.0, 255.0, 129.0, 135.0, 141.0, 155.0, 172.0, 142.0, 142.0,
                141.0, 142.0, 142.0, 171.0, 155.0, 142.0, 135.0, 130.0, 254.0, 172.0, 143.0, 130.0,
                122.0, 129.0, 135.0, 141.0, 155.0, 172.0, 135.0, 139.0, 143.0, 150.0, 155.0, 142.0,
                143.0, 143.0, 143.0, 142.0, 155.0, 150.0, 143.0, 139.0, 135.0, 172.0, 155.0, 142.0,
                135.0, 130.0, 142.0, 142.0, 141.0, 142.0, 142.0, 142.0, 143.0, 143.0, 143.0, 142.0,
                141.0, 143.0, 143.0, 142.0, 141.0, 142.0, 143.0, 143.0, 143.0, 142.0, 143.0, 142.0,
                141.0, 142.0, 142.0, 171.0, 155.0, 142.0, 135.0, 130.0, 155.0, 150.0, 143.0, 139.0,
                135.0, 142.0, 143.0, 143.0, 143.0, 142.0, 135.0, 139.0, 143.0, 150.0, 154.0, 130.0,
                135.0, 143.0, 154.0, 171.0, 254.0, 172.0, 143.0, 130.0, 122.0, 172.0, 155.0, 142.0,
                135.0, 130.0, 143.0, 142.0, 141.0, 142.0, 142.0, 130.0, 135.0, 143.0, 154.0, 171.0,
                122.0, 129.0, 142.0, 172.0, 250.0
            ]
        );
    }

    #[test]
    fn test_extraction_error() {
        let (mut fch, _) = setup();
        let invalid_img = DynamicImage::new_rgb8(0, 0); // Example of an invalid image

        // Expecting a specific error
        match fch.extract(&invalid_img) {
            Err(IFXError::SmallImage) => (),
            Err(_) => panic!("Unexpected error type"),
            Ok(_) => panic!("Expected an error, but extract succeeded"),
        }
    }
}

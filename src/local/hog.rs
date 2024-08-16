use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::hog::{self, HogOptions};

use crate::{
    error::IFXError,
    features::{Extractor, FeatureVector, IFXFeature},
};

use super::LocalFeature;

#[derive(Debug, PartialEq)]
pub struct HOG {
    options: HogOptions,
    img: Option<GrayImage>,
}

impl HOG {
    pub fn new_with_options(options: HogOptions) -> Self {
        Self {
            img: None,
            options: options,
        }
    }
}

impl Extractor for HOG {
    fn extract(&mut self, image: &DynamicImage) -> Result<(), IFXError> {
        if image.height() == 0 || image.width() == 0 {
            return Err(IFXError::SmallImage);
        }

        self.img = Some(image.to_luma8());

        Ok(())
    }
}

impl FeatureVector for HOG {
    fn get_feature_vector(&self) -> Vec<f64> {
        // Retrieve the image buffer
        let img: &ImageBuffer<Luma<u8>, Vec<u8>> = match &self.img {
            Some(img) => img,
            None => panic!("image buffer is null"),
        };

        // Initialize the result as an empty Vec<f64>
        let result: Vec<f64>;

        // Compute HOG descriptors, and handle the result.
        match hog::hog(img, self.options) {
            Ok(buffer) => {
                // Convert each f32 value to f64 and collect into the result vector
                result = buffer.into_iter().map(|v| v as f64).collect();
            }
            Err(error) => {
                // Handle the error by returning an empty Vec<f64>
                eprintln!("{}", error);
                result = Vec::new();
            }
        }

        result
    }
}

impl IFXFeature for HOG {
    fn get_feature_name(&self) -> &str {
        "Histogram of oriented gradients"
    }

    fn get_descriptor_name(&self) -> &str {
        "HOG"
    }
}

impl LocalFeature for HOG {}

#[cfg(test)]
mod tests {

    use super::*;
    use image::DynamicImage;

    // Helper function to initialize the HOG and load the image
    fn setup() -> (HOG, DynamicImage) {
        let hog = HOG::new_with_options(HogOptions::new(
            2,
            false,
            60,
            8,
            1,
        ));
        let img = image::open("./test/01.png")
            .expect("Failed to open the image");
        (hog, img)
    }

    #[test]
    fn test_extract_features() {
        let (mut hog, img) = setup();
        hog.extract(&img).expect("Failed to extract features");

        let vector: Vec<f64> = hog.get_feature_vector();
        
        assert!(!vector.is_empty(), "Feature vector should not be empty");
    }

    #[test]
    fn test_extraction_error() {
        let (mut hog, _) = setup();
        let invalid_img = DynamicImage::new_rgb8(0, 0); // Example of an invalid image

        // Expecting a specific error
        match hog.extract(&invalid_img) {
            Err(IFXError::SmallImage) => (),
            Err(_) => panic!("Unexpected error type"),
            Ok(_) => panic!("Expected an error, but extract succeeded"),
        }
    }
}

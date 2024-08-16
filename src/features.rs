use image::DynamicImage;

use crate::error::IFXError;

pub trait Extractor {
    fn extract(&mut self, image: &DynamicImage) -> Result<(), IFXError>;
}

pub trait FeatureVector {
    fn get_feature_vector(&self) -> Vec<f64>;
}

pub trait IFXFeature : FeatureVector {
    fn get_feature_name(&self) -> &str; // Returns a reference to the feature name

    fn get_descriptor_name(&self) -> &str; // Returns a reference to the descriptor name
}
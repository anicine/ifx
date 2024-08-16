pub mod hog;

use crate::features::{Extractor, IFXFeature};

pub trait LocalFeature: Extractor + IFXFeature {}
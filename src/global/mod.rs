pub mod color_layout;
pub mod edge_histogram;
pub mod fuzzy_color_histogram;

use crate::features::{Extractor, IFXFeature};

pub trait GlobalFeature: Extractor + IFXFeature {}
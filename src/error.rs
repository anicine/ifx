use std::error::Error;
use std::fmt::{Display, Formatter, Result};

/// IFX custom error enum
#[derive(Debug)]
pub enum IFXError {
    NullImage,
    SmallImage,
    /// This variant can hold another error for chaining
    Other(Box<dyn Error + Send + Sync>),
}

impl Display for IFXError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            IFXError::NullImage => write!(f, "Image is null"),
            IFXError::SmallImage => write!(f, "Image is too small"),
            IFXError::Other(err) => write!(f, "Other error: {}", err),
        }
    }
}

// Implementing the std::error::Error trait
impl Error for IFXError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IFXError::Other(err) => Some(err.as_ref()),
            _ => None, // No underlying cause for other errors
        }
    }
}

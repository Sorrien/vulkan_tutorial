use ash::extensions::ext::DebugUtils;
use ash::{vk, Entry};
pub use ash::{Device, Instance};
use std::borrow::Cow;
use std::ffi::CStr;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity > vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );
    }

    vk::FALSE
}

pub fn debug_utils(entry: &Entry, instance: &Instance) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = DebugUtils::new(entry, &instance);

    #[cfg(feature = "validation_layers")]
    let debug_call_back = create_debug_callback(&debug_utils_loader);
    #[cfg(not(feature = "validation_layers"))]
    let debug_call_back = ash::vk::DebugUtilsMessengerEXT::null();

    (debug_utils_loader, debug_call_back)
}

fn create_debug_callback(debug_utils_loader: &DebugUtils) -> ash::vk::DebugUtilsMessengerEXT {
    let debug_info = create_debug_info();

    unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap()
    }
}

pub fn create_debug_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    let x = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback))
        .build();
    x
}

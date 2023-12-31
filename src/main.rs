use vulkan_tutorial::VulkanApplication;

fn main() -> Result<(), winit::error::EventLoopError> {
    let (event_loop, window) = VulkanApplication::init_window(800, 600);

    let mut app = VulkanApplication::new(window);
    app.run(event_loop)
}

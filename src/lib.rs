pub mod buffers;

use std::{
    ffi::{c_char, c_void, CStr},
    fs::File,
    path::Path,
    time::Instant,
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util::read_spv,
    vk::{
        self, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, BlendFactor, BlendOp,
        ColorComponentFlags, ColorSpaceKHR, CommandPoolCreateFlags, ComponentMapping,
        ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags, DescriptorPoolCreateInfo,
        DeviceCreateInfo, DeviceQueueCreateInfo, DynamicState, Extent2D, Format, FrontFace,
        GraphicsPipelineCreateInfo, ImageAspectFlags, ImageLayout, ImageSubresourceRange,
        ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, LogicOp, PhysicalDevice,
        PhysicalDeviceFeatures, PhysicalDeviceType, Pipeline, PipelineBindPoint,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PresentModeKHR, PrimitiveTopology, QueueFlags, RenderPassCreateInfo, SampleCountFlags,
        ShaderStageFlags, SharingMode, SubpassDescription, SurfaceCapabilitiesKHR,
        SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR,
    },
    Device, Entry, Instance,
};
use buffers::{IndexBuffer, MyBuffer, UniformBuffer, UniformBufferObject, Vertex, VertexBuffer};
use glam::{Vec2, Vec3};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub mod debug;

const MAXFRAMESINFLIGHT: usize = 2;

pub struct VulkanApplication {
    window: Window,
    instance: ash::Instance,
    debug_utils: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    surface: vk::SurfaceKHR,
    surface_loader: Surface,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: Swapchain,
    swapchain_images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
    uniform_buffers: Vec<UniformBuffer>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    queue_family_indices: QueueFamilyIndices,
    swapchain_support: SwapchainSupportDetails,
    framebuffer_resized: bool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    start_instant: Instant,
}

impl VulkanApplication {
    pub fn new(window: Window) -> Self {
        let entry = ash::Entry::linked();
        let window_size = window.inner_size();
        let instance = Self::create_instance(&window, &entry);

        let (debug_utils, debug_messenger) = crate::debug::debug_utils(&entry, &instance);

        let surface_loader = Surface::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }
        .expect("failed to create window surface!");

        let (physical_device, queue_family_indices, swapchain_support) =
            Self::pick_physical_device(&instance, &surface_loader, &surface)
                .expect("failed to find physical device!");

        let device =
            Self::create_logical_device(&instance, &physical_device, &queue_family_indices)
                .expect("failed to create logical device!");

        let graphics_queue = unsafe {
            device.get_device_queue(queue_family_indices.graphics_family.unwrap() as u32, 0)
        };

        let present_queue = unsafe {
            device.get_device_queue(queue_family_indices.present_family.unwrap() as u32, 0)
        };

        let (swapchain, swapchain_loader, format, extent) = Self::create_swapchain(
            &instance,
            &device,
            &surface,
            window_size.width,
            window_size.height,
            &queue_family_indices,
            &swapchain_support,
        )
        .expect("failed to create swapchain!");

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .expect("failed to get swapchain images!");

        let swapchain_image_views =
            Self::create_swapchain_image_views(&device, &swapchain_images, format);

        let color_attachment_refs = vec![vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let graphics_subpass =
            Self::create_sub_pass(&color_attachment_refs, PipelineBindPoint::GRAPHICS);

        let render_pass = Self::create_render_pass(&device, &format, vec![graphics_subpass])
            .expect("failed to create render pass!");

        let descriptor_set_layout = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let bindings = [descriptor_set_layout];

        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None) }
                .expect("failed to create descriptor set layout!");

        let descriptor_set_layouts = vec![descriptor_set_layout];

        let (pipeline_layout, graphics_pipeline) =
            Self::create_graphics_pipeline(&device, &render_pass, &descriptor_set_layouts);

        let swapchain_framebuffers =
            Self::create_frame_buffers(&swapchain_image_views, &render_pass, &extent, &device);

        let command_pool = Self::create_command_pool(&device, &queue_family_indices);

        let vertices = vec![
            Vertex::new(Vec2::new(-0.5, -0.5), Vec3::new(1., 0., 0.)),
            Vertex::new(Vec2::new(0.5, -0.5), Vec3::new(0., 1., 0.)),
            Vertex::new(Vec2::new(0.5, 0.5), Vec3::new(0., 0., 1.)),
            Vertex::new(Vec2::new(-0.5, 0.5), Vec3::new(1., 1., 1.)),
        ];

        let vertex_buffer = VertexBuffer::init(
            &instance,
            &physical_device,
            &device,
            &command_pool,
            graphics_queue,
            vertices,
        );

        let indices = vec![0, 1, 2, 2, 3, 0];

        let index_buffer = IndexBuffer::new(
            &instance,
            &physical_device,
            &device,
            &command_pool,
            graphics_queue,
            indices,
        );

        let uniform_buffers = Self::create_uniform_buffers(&instance, &physical_device, &device);

        let descriptor_pool =
            Self::create_descriptor_pool(&device).expect("failed to create descriptor pool!");

        let descriptor_sets =
            Self::create_descriptor_sets(&device, descriptor_pool, &descriptor_set_layouts)
                .expect("failed to allocate descriptor sets!");

        descriptor_sets.iter().zip(&uniform_buffers).for_each(
            |(descriptor_set, uniform_buffer)| {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffer.buffer.buffer)
                    .offset(0)
                    .range(std::mem::size_of::<UniformBufferObject>() as u64);

                let buffer_infos = [buffer_info];
                let write_descriptor_set = vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&buffer_infos);

                let descriptor_writes = [write_descriptor_set];
                let descriptor_copies = [];
                unsafe { device.update_descriptor_sets(&descriptor_writes, &descriptor_copies) };
            },
        );

        let command_buffers = Self::create_command_buffers(&device, &command_pool)
            .expect("failed to allocate command buffers!");

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device);

        Self {
            window,
            instance,
            debug_utils,
            debug_messenger,
            physical_device,
            device,
            surface,
            surface_loader,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_loader,
            swapchain_images,
            format,
            extent,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            vertex_buffer,
            index_buffer,
            uniform_buffers,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            queue_family_indices,
            swapchain_support,
            framebuffer_resized: false,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            start_instant: Instant::now(),
        }
    }

    fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, vk::Result> {
        let descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAXFRAMESINFLIGHT as u32);
        let pool_sizes = [descriptor_pool_size];
        let descriptor_pool_create_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAXFRAMESINFLIGHT as u32);
        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
    }

    fn create_descriptor_sets(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layouts: &Vec<vk::DescriptorSetLayout>,
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
    }

    pub fn run(&mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        self.main_loop(event_loop)
    }

    fn main_loop(&mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        event_loop.run(move |event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                unsafe { self.device.device_wait_idle() }
                    .expect("failed to wait for idle on exit!");
                elwt.exit()
            }
            Event::AboutToWait => {
                //AboutToWait is the new MainEventsCleared
                self.window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id: _,
            } => {
                let size = self.window.inner_size();
                //don't attempt to draw a frame in window size is 0zs
                if size.height > 0 && size.width > 0 {
                    self.draw_frame();
                }
            }
            Event::WindowEvent {
                window_id: _,
                event: WindowEvent::Resized(_new_size),
            } => {
                self.framebuffer_resized = true;
            }
            _ => (),
        })
    }

    fn draw_frame(&mut self) {
        //println!("drawing new frame!");
        unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            )
        }
        .expect("failed to wait for in flight fence!");

        if self.framebuffer_resized {
            self.framebuffer_resized = false;
            self.recreate_swapchain();
            return;
        }

        self.update_uniform_buffer(self.current_frame);

        let (image_index, _was_next_image_acquired) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );

            match result {
                Ok((image_index, was_next_image_acquired)) => {
                    (image_index, was_next_image_acquired)
                }
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        self.framebuffer_resized = false;
                        self.recreate_swapchain();
                        return;
                    }
                    _ => panic!("failed to acquire next swapchain image!"),
                },
            }
        };

        //only reset the fence if we are submitting work
        unsafe {
            self.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
        }
        .expect("failed to reset in flight fence!");

        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[self.current_frame],
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .expect("failed to reset command buffer!");

        self.record_command_buffer(command_buffer, image_index as usize);
        let command_buffers = [command_buffer];
        let command_submit_wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_submit_signal_semaphores =
            [self.render_finished_semaphores[self.current_frame]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&command_submit_wait_semaphores)
            .wait_dst_stage_mask(&wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(&command_submit_signal_semaphores);

        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame],
            )
        }
        .expect("failed to submit draw command buffer!");

        let swapchains = [self.swapchain];
        let present_wait_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&present_wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };
        match present_result {
            Ok(_) => (),
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                    self.framebuffer_resized = false;
                    self.recreate_swapchain();
                    return;
                }
                _ => panic!("failed to present swap chain image!"),
            },
        }

        self.current_frame = (self.current_frame + 1) % MAXFRAMESINFLIGHT;
    }

    fn update_uniform_buffer(&mut self, current_image: usize) {
        //let time = self.start_instant.elapsed().as_millis();
        let now = Instant::now();
        let dur = now - self.start_instant;
        let time = dur.as_millis();//dur.as_millis() / 100;
        let uniform_buffer_object = UniformBufferObject {
            model: glam::Mat4::from_axis_angle(Vec3::Z, time as f32 * 90.),
            view: glam::Mat4::look_at_rh(
                glam::Vec3::new(2., 2., 2.),
                glam::Vec3::new(0., 0., 0.),
                glam::Vec3::Y,
            ),
            proj: glam::Mat4::perspective_rh(
                45.,
                self.extent.width as f32 / self.extent.height as f32,
                0.1,
                10.,
            ),
        };

        let uniform_buffer = &mut self.uniform_buffers[current_image];
        uniform_buffer.modify_buffer(vec![uniform_buffer_object]);
    }

    fn recreate_swapchain(&mut self) {
        //may want to look into using the old swapchain to create a new one instead of just waiting for idle then destroying the current one.
        unsafe { self.device.device_wait_idle() }.expect("failed to wait for device idle!");

        self.cleanup_swapchain();

        let swapchain_details = SwapchainSupportDetails::new(
            &self.physical_device,
            &self.surface_loader,
            &self.surface,
        );

        self.swapchain_support = swapchain_details;

        let window_size = self.window.inner_size();

        let (swapchain, swapchain_loader, format, extent) = Self::create_swapchain(
            &self.instance,
            &self.device,
            &self.surface,
            window_size.width,
            window_size.height,
            &self.queue_family_indices,
            &self.swapchain_support,
        )
        .expect("failed to recreate swapchain!");
        self.swapchain = swapchain;
        self.swapchain_loader = swapchain_loader;
        self.format = format;
        self.extent = extent;

        self.swapchain_images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .expect("failed to get swapchain images!");

        self.swapchain_image_views =
            Self::create_swapchain_image_views(&self.device, &self.swapchain_images, format);

        self.swapchain_framebuffers = Self::create_frame_buffers(
            &self.swapchain_image_views,
            &self.render_pass,
            &self.extent,
            &self.device,
        );
    }

    fn create_instance(window: &Window, entry: &Entry) -> Instance {
        #[cfg(feature = "validation_layers")]
        let enable_validation_layers = true;
        #[cfg(not(feature = "validation_layers"))]
        let enable_validation_layers = false;

        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello Triangle\0") };
        let engine_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"No Engine\0") };

        let mut required_extension_names = vec![];

        let mut window_extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();

        required_extension_names.append(&mut window_extension_names);
        #[cfg(feature = "validation_layers")]
        required_extension_names.push(DebugUtils::NAME.as_ptr());
        #[cfg(feature = "validation_layers")]
        println!("Validation Layers enabled!");

        let extension_properties = unsafe { entry.enumerate_instance_extension_properties(None) }
            .expect("failed to enumerate instance extension props!");

        println!("Enabled extensions:");
        for extension_name in required_extension_names.iter() {
            let str = unsafe { CStr::from_ptr(*extension_name) }
                .to_str()
                .expect("failed to get ext name str");

            if extension_properties.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required extensions were not available!");
            }
        }
        println!("");

        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_extension_names(&required_extension_names);

        let debug_info = debug::create_debug_info();
        if enable_validation_layers {
            Self::check_validation_layer_support(&entry, &layers_names_raw);
            create_info = create_info.enabled_layer_names(&layers_names_raw);

            create_info.p_next =
                &debug_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let instance: Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error")
        };

        instance
    }

    fn check_validation_layer_support(entry: &Entry, layers_names_raw: &Vec<*const c_char>) {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("failed to get available layers!");

        for name in layers_names_raw.iter() {
            let str = unsafe { CStr::from_ptr(*name) }
                .to_str()
                .expect("failed to get layer name str");

            if available_layers.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.layer_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required layers were not available!");
            }
        }
    }

    pub fn init_window(width: u32, height: u32) -> (EventLoop<()>, winit::window::Window) {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title("Vulkan Tutorial")
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)
            .unwrap();

        (event_loop, window)
    }

    fn pick_physical_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: &SurfaceKHR,
    ) -> Option<(PhysicalDevice, QueueFamilyIndices, SwapchainSupportDetails)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("failed to find physical devices!");

        if physical_devices.len() == 0 {
            panic!("failed to find GPUs with Vulkan support!");
        }

        let mut scored_devices = physical_devices
            .iter()
            .filter_map(|device| {
                if let Some((score, queue_family_indices, swapchain_details)) =
                    Self::rate_device(instance, device, surface_loader, surface)
                {
                    Some((score, device, queue_family_indices, swapchain_details))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        scored_devices.sort_by(|(a, _, _, _), (b, _, _, _)| a.cmp(b));

        if let Some((_, device, queue_family_indices, swapchain_details)) = scored_devices.last() {
            Some((**device, *queue_family_indices, swapchain_details.clone()))
        } else {
            None
        }
    }

    fn rate_device(
        instance: &Instance,
        device: &PhysicalDevice,
        surface_loader: &Surface,
        surface: &SurfaceKHR,
    ) -> Option<(u32, QueueFamilyIndices, SwapchainSupportDetails)> {
        let mut score = 0;

        let features = unsafe { instance.get_physical_device_features(*device) };
        let properties = unsafe { instance.get_physical_device_properties(*device) };

        if properties.device_type == PhysicalDeviceType::DISCRETE_GPU {
            // Discrete GPUs have a significant performance advantage
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += properties.limits.max_image_dimension2_d;

        let device_props = unsafe { instance.get_physical_device_properties(*device) };
        let device_name =
            unsafe { CStr::from_ptr(device_props.device_name.as_ptr()).to_str() }.unwrap();
        // Application can't function without geometry shaders or the graphics queue family
        if features.geometry_shader == 1 {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(*device) };

            let queue_family_indices =
                QueueFamilyIndices::new(&queue_families, surface_loader, device, surface);
            if queue_family_indices.graphics_family.is_some() {
                let swapchain_details =
                    SwapchainSupportDetails::new(device, surface_loader, surface);

                if swapchain_details.formats.len() > 0 && swapchain_details.present_modes.len() > 0
                {
                    Some((score, queue_family_indices, swapchain_details))
                } else {
                    println!(
                        "failed to find swapchain format or present mode on physical device! {}",
                        device_name
                    );
                    None
                }
            } else {
                println!(
                    "failed to find graphics family queue on physical device! {}",
                    device_name
                );
                None
            }
        } else {
            None
        }
    }

    fn create_logical_device(
        instance: &Instance,
        device: &PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
    ) -> Result<ash::Device, vk::Result> {
        let queue_create_infos = [queue_family_indices.graphics_family.unwrap() as u32].map(|i| {
            DeviceQueueCreateInfo::default()
                .queue_family_index(i)
                .queue_priorities(&[1.])
        });

        let device_features = PhysicalDeviceFeatures::default().geometry_shader(true);

        //may want to add check against available extensions later but the availability of present implies swap chain extension availability.
        let device_extension_names_raw = [
            Swapchain::NAME.as_ptr(),
            //#[cfg(any(target_os = "macos", target_os = "ios"))]
            //KhrPortabilitySubsetFn::NAME.as_ptr(),
        ];
        let device_create_info = DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extension_names_raw); //device specific layers are outdated but keep in mind if an issue crops up on older hardware

        unsafe { instance.create_device(*device, &device_create_info, None) }
    }

    fn create_swapchain(
        instance: &Instance,
        logical_device: &Device,
        surface: &SurfaceKHR,
        window_width: u32,
        window_height: u32,
        queue_family_indices: &QueueFamilyIndices,
        swapchain_support: &SwapchainSupportDetails,
    ) -> Result<(vk::SwapchainKHR, Swapchain, Format, Extent2D), vk::Result> {
        let surface_format = Self::choose_swapchain_surface_format(&swapchain_support.formats)
            .expect("failed to find surface format!");
        let present_mode = Self::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent =
            Self::choose_swap_extent(&swapchain_support.capabilities, window_width, window_height);

        let image_count = swapchain_support.capabilities.min_image_count + 1;

        //if max_image_count is 0 then there is no max
        let image_count = if swapchain_support.capabilities.max_image_count > 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
            swapchain_support.capabilities.max_image_count
        } else {
            image_count
        };

        let (sharing_mode, queue_indices) =
            if queue_family_indices.graphics_family != queue_family_indices.present_family {
                (
                    SharingMode::CONCURRENT,
                    vec![
                        queue_family_indices.graphics_family.unwrap() as u32,
                        queue_family_indices.present_family.unwrap() as u32,
                    ],
                )
            } else {
                (SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_loader = Swapchain::new(instance, logical_device);
        let swapchain_create_info = SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        Ok((swapchain, swapchain_loader, surface_format.format, extent))
    }

    fn choose_swapchain_surface_format(
        available_formats: &Vec<SurfaceFormatKHR>,
    ) -> Option<&SurfaceFormatKHR> {
        if let Some(desired_format) = available_formats.iter().find(|format| {
            format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                && format.format == Format::B8G8R8A8_SRGB
        }) {
            Some(desired_format)
        } else {
            available_formats.first()
        }
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &Vec<PresentModeKHR>,
    ) -> PresentModeKHR {
        let desired_mode = PresentModeKHR::MAILBOX;
        let is_desired_mode_available = available_present_modes
            .iter()
            .any(|present_mode| *present_mode == desired_mode);
        if is_desired_mode_available {
            desired_mode
        } else {
            PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        capabilities: &SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> Extent2D {
        match capabilities.current_extent.width {
            //the max value of u32 is a special value to indicate that we must choose a resolution with the current min and max extents
            //should look into how DPI scaling is handled by winit and if this is the pixel extent or if this includes dpi scaling.
            u32::MAX => vk::Extent2D {
                width: window_width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            },
            _ => capabilities.current_extent,
        }
    }

    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: Format,
    ) -> Vec<ImageView> {
        let component_mapping = ComponentMapping::default()
            .r(ComponentSwizzle::IDENTITY)
            .g(ComponentSwizzle::IDENTITY)
            .b(ComponentSwizzle::IDENTITY)
            .a(ComponentSwizzle::IDENTITY);
        let subresource_range = ImageSubresourceRange::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        swapchain_images
            .iter()
            .map(|image| {
                ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(ImageViewType::TYPE_2D)
                    .format(swapchain_image_format)
                    .components(component_mapping)
                    .subresource_range(subresource_range)
            })
            .map(|image_view_create_info| {
                unsafe { device.create_image_view(&image_view_create_info, None) }
                    .expect("failed to create image view for swapchain image!")
            })
            .collect::<Vec<_>>()
    }

    fn create_shader_module<P>(path: P, device: &Device) -> vk::ShaderModule
    where
        P: AsRef<Path>,
    {
        let mut spv_file = File::open(path).unwrap();
        let shader_code = read_spv(&mut spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };
        shader_module
    }

    fn create_graphics_pipeline(
        device: &Device,
        render_pass: &vk::RenderPass,
        descriptor_set_layouts: &Vec<vk::DescriptorSetLayout>,
    ) -> (PipelineLayout, Pipeline) {
        let vertex_shader_module =
            Self::create_shader_module("shaders/triangle/triangle.vert.spv", device);
        let fragment_shader_module =
            Self::create_shader_module("shaders/triangle/triangle.frag.spv", device);
        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let vertex_shader_stage_info = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(shader_entry_name);

        let fragment_shader_stage_info = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(shader_entry_name);

        let shader_stages = vec![vertex_shader_stage_info, fragment_shader_stage_info];

        let dynamic_state = PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]);

        let vertex_attrib_descs = Vertex::get_attribute_descriptions();
        let vertex_bind_desc = Vertex::get_binding_description();
        let vertex_bind_descs = [vertex_bind_desc];

        let vertex_input_info = PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_bind_descs)
            .vertex_attribute_descriptions(&vertex_attrib_descs);

        let input_assembly = PipelineInputAssemblyStateCreateInfo::default()
            .topology(PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlags::TYPE_1);

        let color_blend_attachment = PipelineColorBlendAttachmentState::default()
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD);

        let color_blend_attachments = [color_blend_attachment];
        let color_blending = PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&color_blend_attachments);

        let pipeline_layout_info =
            PipelineLayoutCreateInfo::default().set_layouts(descriptor_set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

        let pipeline_info = GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            //.depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0);

        let graphics_pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .expect("failed to create graphics pipeline!");

        unsafe {
            device.destroy_shader_module(fragment_shader_module, None);
            device.destroy_shader_module(vertex_shader_module, None);
        }

        (pipeline_layout, *graphics_pipelines.first().unwrap())
    }

    fn create_sub_pass(
        color_attachment_refs: &Vec<vk::AttachmentReference>,
        pipeline_bind_point: vk::PipelineBindPoint,
    ) -> vk::SubpassDescription {
        let subpass = vk::SubpassDescription::default()
            .color_attachments(&color_attachment_refs)
            .pipeline_bind_point(pipeline_bind_point);
        subpass
    }

    fn create_render_pass(
        device: &Device,
        swapchain_image_format: &Format,
        subpasses: Vec<SubpassDescription>,
    ) -> Result<vk::RenderPass, vk::Result> {
        let color_attachment = AttachmentDescription::default()
            .format(*swapchain_image_format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR);

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let color_attachments = [color_attachment];

        let render_pass_info = RenderPassCreateInfo::default()
            .attachments(&color_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        unsafe { device.create_render_pass(&render_pass_info, None) }
    }

    fn create_frame_buffers(
        swapchain_image_views: &Vec<ImageView>,
        render_pass: &vk::RenderPass,
        swapchain_extent: &Extent2D,
        device: &Device,
    ) -> Vec<vk::Framebuffer> {
        swapchain_image_views
            .iter()
            .map(|swapchain_image_view| {
                let attachments = [*swapchain_image_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(*render_pass)
                    .attachments(&attachments)
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None) }
                    .expect("failed to create framebuffer!")
            })
            .collect::<Vec<_>>()
    }

    fn create_command_pool(
        device: &Device,
        queue_family_indices: &QueueFamilyIndices,
    ) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_family.unwrap() as u32);

        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .expect("failed to create command pool!");
        command_pool
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAXFRAMESINFLIGHT as u32);

        unsafe { device.allocate_command_buffers(&alloc_info) }
    }

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .expect("failed to begin recording command buffer!");

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            )
        };

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .min_depth(0.)
            .max_depth(1.);
        unsafe { self.device.cmd_set_viewport(command_buffer, 0, &[viewport]) };

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.extent);
        unsafe { self.device.cmd_set_scissor(command_buffer, 0, &[scissor]) };

        let vertex_buffers = [self.vertex_buffer.buffer.buffer];
        let offsets = [0];
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets)
        }
        unsafe {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer.buffer,
                0,
                vk::IndexType::UINT32,
            )
        };
        unsafe { self.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &self.descriptor_sets, &[]) };
        unsafe {
            self.device.cmd_draw_indexed(
                command_buffer,
                self.index_buffer.index_count as u32,
                1,
                0,
                0,
                0,
            )
        };

        unsafe { self.device.cmd_end_render_pass(command_buffer) };

        unsafe { self.device.end_command_buffer(command_buffer) }
            .expect("failed to record command buffer!");
    }

    fn create_sync_objects(
        device: &Device,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let max_frames_range = 0..MAXFRAMESINFLIGHT;

        let image_available_semaphores = max_frames_range
            .clone()
            .map(|_| {
                unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .expect("failed to create semaphore!")
            })
            .collect::<Vec<_>>();

        let render_finished_semaphores = max_frames_range
            .clone()
            .map(|_| {
                unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .expect("failed to create semaphore!")
            })
            .collect::<Vec<_>>();
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let in_flight_fences = max_frames_range
            .clone()
            .map(|_| {
                unsafe { device.create_fence(&fence_create_info, None) }
                    .expect("failed to create fence!")
            })
            .collect::<Vec<_>>();

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.swapchain_framebuffers
                .iter()
                .for_each(|framebuffer| self.device.destroy_framebuffer(*framebuffer, None));
            self.swapchain_image_views
                .iter()
                .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn create_uniform_buffers(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
    ) -> Vec<UniformBuffer> {
        (0..MAXFRAMESINFLIGHT)
            .map(|_| UniformBuffer::new(instance, physical_device, device, 1))
            .collect::<Vec<UniformBuffer>>()
    }
}

impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAXFRAMESINFLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.cleanup_swapchain();
            for i in 0..MAXFRAMESINFLIGHT {
                let uniform_buffer = &mut self.uniform_buffers[i];
                uniform_buffer.cleanup(&self.device);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.index_buffer.cleanup(&self.device);
            self.vertex_buffer.cleanup(&self.device);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            #[cfg(feature = "validation_layers")]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: Option<usize>,
    present_family: Option<usize>,
}

impl QueueFamilyIndices {
    fn new(
        queue_families: &Vec<vk::QueueFamilyProperties>,
        surface_loader: &Surface,
        pdevice: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        let graphics_family_index = if let Some((graphics_family_index, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, queue)| queue.queue_flags.contains(QueueFlags::GRAPHICS))
        {
            Some(graphics_family_index)
        } else {
            None
        };

        let present_family_index = if let Some((present_family_index, _)) =
            queue_families.iter().enumerate().find(|(i, _)| {
                unsafe {
                    surface_loader
                        .get_physical_device_surface_support(*pdevice, *i as u32, *surface)
                }
                .unwrap()
            }) {
            Some(present_family_index)
        } else {
            None
        };

        Self {
            graphics_family: graphics_family_index,
            present_family: present_family_index,
        }
    }
}

#[derive(Clone)]
struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn new(
        device: &vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        let capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(*device, *surface) }
                .expect("failed to get surface capabilites!");
        let formats =
            unsafe { surface_loader.get_physical_device_surface_formats(*device, *surface) }
                .expect("failed to get device surface formats!");
        let present_modes =
            unsafe { surface_loader.get_physical_device_surface_present_modes(*device, *surface) }
                .expect("failed to get device surface present modes!");

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }
}

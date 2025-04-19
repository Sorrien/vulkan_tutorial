use std::{
    ffi::{c_char, c_void, CStr},
    fs::File,
    path::Path,
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util::read_spv,
    vk::{
        self, AttachmentLoadOp, AttachmentStoreOp, BlendFactor, BlendOp, ColorComponentFlags,
        ColorSpaceKHR, CompositeAlphaFlagsKHR, DescriptorPoolCreateInfo, DeviceCreateInfo,
        DeviceQueueCreateInfo, DynamicState, Extent2D, Format, ImageLayout, ImageUsageFlags,
        ImageView, LogicOp, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceType,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PresentModeKHR,
        PrimitiveTopology, RenderPassCreateInfo, SampleCountFlags, ShaderStageFlags, SharingMode,
        SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR,
    },
    Entry, Instance,
};
use winit::window::Window;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{
    buffers::{
        begin_single_time_commands, copy_buffer_to_image, copy_to_staging_buffer,
        end_single_time_commands, find_memory_type, MyBuffer, UniformBuffer, Vertex,
    },
    debug, MAXFRAMESINFLIGHT,
};

pub struct BaseVulkanState {
    pub window: Window,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_indices: QueueFamilyIndices,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub swapchain_support: SwapchainSupportDetails,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub debug_utils: DebugUtils,
    msaa_samples: SampleCountFlags,
}

impl BaseVulkanState {
    pub fn new(window: Window) -> Self {
        let entry = ash::Entry::linked();
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

        let (physical_device, queue_family_indices, swapchain_support, msaa_samples) =
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

        Self {
            window,
            instance,
            physical_device,
            device,
            queue_family_indices,
            graphics_queue,
            present_queue,
            swapchain_support,
            surface,
            surface_loader,
            debug_utils,
            debug_messenger,
            msaa_samples,
        }
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

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
    ) -> Option<(
        vk::PhysicalDevice,
        QueueFamilyIndices,
        SwapchainSupportDetails,
        vk::SampleCountFlags,
    )> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("failed to find physical devices!");

        if physical_devices.len() == 0 {
            panic!("failed to find GPUs with Vulkan support!");
        }

        let mut scored_devices = physical_devices
            .iter()
            .filter_map(|device| {
                if let Some((score, queue_family_indices, swapchain_details, msaa_samples)) =
                    Self::rate_device(instance, device, surface_loader, surface)
                {
                    Some((
                        score,
                        device,
                        queue_family_indices,
                        swapchain_details,
                        msaa_samples,
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        scored_devices.sort_by(|(a, _, _, _, _), (b, _, _, _, _)| a.cmp(b));

        if let Some((_, device, queue_family_indices, swapchain_details, msaa_samples)) =
            scored_devices.last()
        {
            Some((
                **device,
                *queue_family_indices,
                swapchain_details.clone(),
                *msaa_samples,
            ))
        } else {
            None
        }
    }

    fn rate_device(
        instance: &Instance,
        device: &PhysicalDevice,
        surface_loader: &Surface,
        surface: &SurfaceKHR,
    ) -> Option<(
        u32,
        QueueFamilyIndices,
        SwapchainSupportDetails,
        vk::SampleCountFlags,
    )> {
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
        // Application can't function without geometry shaders or the graphics queue family or anisotropy (we could remove anisotropy)
        if features.geometry_shader == 1 && features.sampler_anisotropy == 1 {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(*device) };

            let queue_family_indices =
                QueueFamilyIndices::new(&queue_families, surface_loader, device, surface);
            if queue_family_indices.graphics_family.is_some() {
                let swapchain_details =
                    SwapchainSupportDetails::new(device, surface_loader, surface);

                if swapchain_details.formats.len() > 0 && swapchain_details.present_modes.len() > 0
                {
                    let msaa_samples = Self::get_max_usable_sample_count(device_props);
                    Some((score, queue_family_indices, swapchain_details, msaa_samples))
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

    pub fn get_max_usable_sample_count(
        physical_device_properties: vk::PhysicalDeviceProperties,
    ) -> vk::SampleCountFlags {
        let counts = physical_device_properties
            .limits
            .framebuffer_color_sample_counts
            & physical_device_properties
                .limits
                .framebuffer_depth_sample_counts;

        let sorted_desired_sample_counts = [
            vk::SampleCountFlags::TYPE_64,
            vk::SampleCountFlags::TYPE_32,
            vk::SampleCountFlags::TYPE_16,
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ];

        let mut allowed_sample_counts = sorted_desired_sample_counts
            .iter()
            .filter(|desired_sample_count| counts.contains(**desired_sample_count));

        if let Some(sample_count) = allowed_sample_counts.next() {
            *sample_count
        } else {
            vk::SampleCountFlags::TYPE_1
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

        let device_features = PhysicalDeviceFeatures::default()
            .geometry_shader(true)
            .sampler_anisotropy(true)
            .sample_rate_shading(true);

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

    pub fn create_swapchain(
        &self,
        window_width: u32,
        window_height: u32,
    ) -> Result<(vk::SwapchainKHR, Swapchain, Format, Extent2D), vk::Result> {
        let surface_format = Self::choose_swapchain_surface_format(&self.swapchain_support.formats)
            .expect("failed to find surface format!");
        let present_mode =
            Self::choose_swapchain_present_mode(&self.swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(
            &self.swapchain_support.capabilities,
            window_width,
            window_height,
        );

        let image_count = self.swapchain_support.capabilities.min_image_count + 1;

        //if max_image_count is 0 then there is no max
        let image_count = if self.swapchain_support.capabilities.max_image_count > 0
            && image_count > self.swapchain_support.capabilities.max_image_count
        {
            self.swapchain_support.capabilities.max_image_count
        } else {
            image_count
        };

        let (sharing_mode, queue_indices) = if self.queue_family_indices.graphics_family
            != self.queue_family_indices.present_family
        {
            (
                SharingMode::CONCURRENT,
                vec![
                    self.queue_family_indices.graphics_family.unwrap() as u32,
                    self.queue_family_indices.present_family.unwrap() as u32,
                ],
            )
        } else {
            (SharingMode::EXCLUSIVE, vec![])
        };

        let swapchain_loader = Swapchain::new(&self.instance, &self.device);
        let swapchain_create_info = SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(self.swapchain_support.capabilities.current_transform)
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

    pub fn create_swapchain_image_views(
        &self,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: Format,
    ) -> Vec<ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                self.create_image_view(
                    *image,
                    swapchain_image_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn create_render_pass(
        &self,
        subpasses: Vec<vk::SubpassDescription>,
        swapchain_format: vk::Format,
    ) -> Result<vk::RenderPass, vk::Result> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(swapchain_format)
            .samples(self.msaa_samples)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescription::default()
            .format(self.find_depth_format())
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescription::default()
            .format(swapchain_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let attachments = [color_attachment, depth_attachment, color_attachment_resolve];

        let render_pass_info = RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        unsafe { self.device.create_render_pass(&render_pass_info, None) }
    }

    pub fn create_descriptor_set_layout(&self) -> Vec<vk::DescriptorSetLayout> {
        let descriptor_set_layout = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [descriptor_set_layout, sampler_layout_binding];

        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        }
        .expect("failed to create descriptor set layout!");

        let descriptor_set_layouts = vec![descriptor_set_layout];
        descriptor_set_layouts
    }

    pub fn create_graphics_pipeline(
        &self,
        render_pass: &vk::RenderPass,
        descriptor_set_layouts: &Vec<vk::DescriptorSetLayout>,
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let vertex_shader_module = self.create_shader_module("shaders/triangle/triangle.vert.spv");
        let fragment_shader_module =
            self.create_shader_module("shaders/triangle/triangle.frag.spv");
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
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(true)
            .rasterization_samples(self.msaa_samples);

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

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0);

        let graphics_pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .expect("failed to create graphics pipeline!");

        unsafe {
            self.device
                .destroy_shader_module(fragment_shader_module, None);
            self.device
                .destroy_shader_module(vertex_shader_module, None);
        }

        (pipeline_layout, *graphics_pipelines.first().unwrap())
    }

    fn create_shader_module<P>(&self, path: P) -> vk::ShaderModule
    where
        P: AsRef<Path>,
    {
        let mut spv_file = File::open(path).unwrap();
        let shader_code = read_spv(&mut spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };
        shader_module
    }

    pub fn create_uniform_buffers(&self) -> Vec<UniformBuffer> {
        (0..MAXFRAMESINFLIGHT)
            .map(|_| UniformBuffer::new(&self.instance, &self.physical_device, &self.device, 1))
            .collect::<Vec<UniformBuffer>>()
    }

    pub fn create_descriptor_pool(&self) -> Result<vk::DescriptorPool, vk::Result> {
        let uniform_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAXFRAMESINFLIGHT as u32);

        let sampler_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAXFRAMESINFLIGHT as u32);

        let pool_sizes = [uniform_descriptor_pool_size, sampler_descriptor_pool_size];
        let descriptor_pool_create_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAXFRAMESINFLIGHT as u32);
        unsafe {
            self.device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
        }
    }

    pub fn create_descriptor_sets(
        &self,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layouts: &Vec<vk::DescriptorSetLayout>,
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
        }
    }

    pub fn create_command_pool(&self) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.queue_family_indices.graphics_family.unwrap() as u32);

        let command_pool = unsafe { self.device.create_command_pool(&pool_info, None) }
            .expect("failed to create command pool!");
        command_pool
    }

    pub fn create_command_buffers(
        &self,
        command_pool: &vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAXFRAMESINFLIGHT as u32);

        unsafe { self.device.allocate_command_buffers(&alloc_info) }
    }

    pub fn create_texture_image<P>(
        &self,
        command_pool: &vk::CommandPool,
        path: P,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView, u32)
    where
        P: AsRef<Path>,
    {
        let img = image::io::Reader::open(path)
            .expect("failed to load image!")
            .decode()
            .unwrap()
            .to_rgba8();

        let (width, height) = img.dimensions();
        let image_extent = vk::Extent2D { width, height };
        let size = (std::mem::size_of::<u8>() * img.len()) as u64;
        let image_data = img.into_raw();
        let mut staging_buffer = MyBuffer::init(
            &self.instance,
            &self.physical_device,
            &self.device,
            size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        copy_to_staging_buffer::<u8>(&self.device, &staging_buffer, size, &image_data);

        let format = vk::Format::R8G8B8A8_SRGB;
        let mip_levels = (image_extent.width.max(image_extent.height) as f32)
            .log2()
            .floor() as u32;
        let (texture_image, texture_image_memory) = self.create_image(
            image_extent,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
        );

        self.transition_image_layout(
            command_pool,
            texture_image,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );
        copy_buffer_to_image(
            &self.device,
            command_pool,
            self.graphics_queue,
            staging_buffer.buffer,
            texture_image,
            image_extent,
        );
        self.transition_image_layout(
            command_pool,
            texture_image,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );

        self.generate_mipmaps(
            command_pool,
            texture_image,
            format,
            image_extent,
            mip_levels,
        );

        staging_buffer.cleanup(&self.device);

        let texture_image_view = self.create_texture_image_view(texture_image, mip_levels);

        (
            texture_image,
            texture_image_memory,
            texture_image_view,
            mip_levels,
        )
    }

    pub fn transition_image_layout(
        &self,
        command_pool: &vk::CommandPool,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let command_buffer = begin_single_time_commands(&self.device, command_pool);

        let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) = if old_layout
            == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        } else if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        {
            (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
        } else {
            panic!("unsupported layout transition!");
        };

        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            let depth = vk::ImageAspectFlags::DEPTH;
            if Self::has_stencil_component(format) {
                depth | vk::ImageAspectFlags::STENCIL
            } else {
                depth
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(0)
            .dst_queue_family_index(0)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        let image_memory_barriers = [barrier];
        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barriers,
            )
        }

        end_single_time_commands(
            &self.device,
            command_pool,
            self.graphics_queue,
            command_buffer,
        );
    }

    pub fn create_image(
        &self,
        img_extent: Extent2D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(img_extent.into())
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(num_samples);

        let image = unsafe { self.device.create_image(&image_info, None) }
            .expect("failed to create image!");

        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(find_memory_type(
                &self.instance,
                &self.physical_device,
                mem_requirements.memory_type_bits,
                properties,
            ));

        let image_memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
            .expect("failed to allocate image memory!");

        unsafe { self.device.bind_image_memory(image, image_memory, 0) }
            .expect("failed to bind image memory!");

        (image, image_memory)
    }

    pub fn generate_mipmaps(
        &self,
        command_pool: &vk::CommandPool,
        image: vk::Image,
        format: vk::Format,
        image_extent: Extent2D,
        mip_levels: u32,
    ) {
        let format_properties = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device, format)
        };

        if !(format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
        {
            panic!("texture image format does not support linear bitting!");
        }

        let command_buffer = begin_single_time_commands(&self.device, command_pool);
        let aspect_mask = vk::ImageAspectFlags::COLOR;

        let mut barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut mip_width = image_extent.width;
        let mut mip_height = image_extent.height;

        for i in 1..mip_levels {
            let src_mip_level = i - 1;
            let dst_mip_level = i;
            barrier.subresource_range.base_mip_level = src_mip_level;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;

            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            let image_memory_barriers = [barrier];
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barriers,
                )
            }

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: mip_width as i32,
                        y: mip_height as i32,
                        z: 1,
                    },
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect_mask)
                        .mip_level(src_mip_level)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: if mip_width > 1 {
                            (mip_width / 2) as i32
                        } else {
                            1
                        },
                        y: if mip_height > 1 {
                            (mip_height / 2) as i32
                        } else {
                            1
                        },
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect_mask)
                        .mip_level(dst_mip_level)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let regions = [blit];
            unsafe {
                self.device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                    vk::Filter::LINEAR,
                )
            }

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
            let image_memory_barriers = [barrier];

            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barriers,
                )
            };

            if mip_width > 1 {
                mip_width /= 2;
            }

            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
        let image_memory_barriers = [barrier];

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barriers,
            )
        };

        end_single_time_commands(
            &self.device,
            command_pool,
            self.graphics_queue,
            command_buffer,
        );
    }

    pub fn create_texture_image_view(&self, image: vk::Image, mip_levels: u32) -> ImageView {
        self.create_image_view(
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> ImageView {
        let component_mapping = vk::ComponentMapping::default();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(component_mapping)
            .subresource_range(subresource_range);

        let image_view = unsafe { self.device.create_image_view(&image_view_create_info, None) }
            .expect("failed to create image view!");
        image_view
    }

    pub fn create_texture_sampler(&self, max_anisotropy: f32, mip_levels: u32) -> vk::Sampler {
        let filter = vk::Filter::LINEAR;
        let address_mode = vk::SamplerAddressMode::REPEAT;
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
            .anisotropy_enable(true)
            .max_anisotropy(max_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(mip_levels as f32);

        unsafe { self.device.create_sampler(&sampler_info, None) }
            .expect("failed to create texture sampler!")
    }

    pub fn create_depth_resources(
        &self,
        extent: Extent2D,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let depth_format = self.find_depth_format();
        let mip_levels = 1;

        let (depth_image, depth_image_memory) = self.create_image(
            extent,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mip_levels,
            self.msaa_samples,
        );
        let depth_image_view = self.create_image_view(
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
            mip_levels,
        );

        (depth_image, depth_image_memory, depth_image_view)
    }

    pub fn find_depth_format(&self) -> vk::Format {
        self.find_supported_format(
            vec![
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    pub fn find_supported_format(
        &self,
        candidates: Vec<vk::Format>,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        *candidates
            .iter()
            .filter(|candidate| {
                let format_properties = unsafe {
                    self.instance
                        .get_physical_device_format_properties(self.physical_device, **candidate)
                };
                if tiling == vk::ImageTiling::LINEAR
                    && (format_properties.linear_tiling_features & features) == features
                {
                    true
                } else if tiling == vk::ImageTiling::OPTIMAL
                    && (format_properties.optimal_tiling_features & features) == features
                {
                    true
                } else {
                    false
                }
            })
            .next()
            .unwrap()
    }

    pub fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    pub fn create_color_resources(
        &self,
        format: vk::Format,
        extent: Extent2D,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let (image, image_memory) = self.create_image(
            extent,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            self.msaa_samples,
        );
        let image_view = self.create_image_view(image, format, vk::ImageAspectFlags::COLOR, 1);

        (image, image_memory, image_view)
    }
}

impl Drop for BaseVulkanState {
    fn drop(&mut self) {
        unsafe {
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
pub struct QueueFamilyIndices {
    pub graphics_family: Option<usize>,
    pub present_family: Option<usize>,
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
            .find(|(_, queue)| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
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
pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
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

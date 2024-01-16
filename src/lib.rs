use std::time::Instant;

use ash::{
    extensions::khr::Swapchain,
    vk::{self, Extent2D, ImageView, PipelineBindPoint},
    Device,
};
use buffers::{IndexBuffer, UniformBuffer, UniformBufferObject, VertexBuffer};
use glam::Vec3;
use vulkan_base::{BaseVulkanState, SwapchainSupportDetails};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub mod buffers;
pub mod debug;
pub mod models;
pub mod vulkan_base;

const MAXFRAMESINFLIGHT: usize = 2;

pub struct VulkanApplication {
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
    framebuffer_resized: bool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    start_instant: Instant,
    texture_image_memory: vk::DeviceMemory,
    texture_image: vk::Image,
    texture_image_view: ImageView,
    texture_sampler: vk::Sampler,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: Swapchain,
    swapchain_images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    base_vulkan_state: BaseVulkanState,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: ImageView,
}

impl VulkanApplication {
    pub fn new(window: Window) -> Self {
        let base_vulkan_state = BaseVulkanState::new(window);

        let window_size = base_vulkan_state.window.inner_size();
        let (swapchain, swapchain_loader, format, extent) = base_vulkan_state
            .create_swapchain(window_size.width, window_size.height)
            .expect("failed to create swapchain!");

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .expect("failed to get swapchain images!");

        let swapchain_image_views =
            base_vulkan_state.create_swapchain_image_views(&swapchain_images, format);

        let color_attachment_refs = vec![vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let graphics_subpass = Self::create_sub_pass(
            &color_attachment_refs,
            &depth_attachment_ref,
            PipelineBindPoint::GRAPHICS,
        );

        let render_pass = base_vulkan_state
            .create_render_pass(vec![graphics_subpass], format)
            .expect("failed to create render pass!");

        let descriptor_set_layouts = base_vulkan_state.create_descriptor_set_layout();

        let (pipeline_layout, graphics_pipeline) =
            base_vulkan_state.create_graphics_pipeline(&render_pass, &descriptor_set_layouts);

        let (depth_image, depth_image_memory, depth_image_view) =
            base_vulkan_state.create_depth_resources(extent);

        let swapchain_framebuffers = Self::create_frame_buffers(
            &swapchain_image_views,
            &render_pass,
            &extent,
            &base_vulkan_state.device,
            depth_image_view,
        );

        let command_pool = base_vulkan_state.create_command_pool();

        let (texture_image, texture_image_memory) =
            base_vulkan_state.create_texture_image(&command_pool, "textures/viking_room.png");

        let physical_device_properties = unsafe {
            base_vulkan_state
                .instance
                .get_physical_device_properties(base_vulkan_state.physical_device)
        };

        let texture_image_view = base_vulkan_state.create_texture_image_view(texture_image);

        let texture_sampler = base_vulkan_state
            .create_texture_sampler(physical_device_properties.limits.max_sampler_anisotropy);

        let (vertices, indices) = models::load_model("models/viking_room.obj");

        let vertex_buffer = VertexBuffer::init(
            &base_vulkan_state.instance,
            &base_vulkan_state.physical_device,
            &base_vulkan_state.device,
            &command_pool,
            base_vulkan_state.graphics_queue,
            vertices,
        );

        let index_buffer = IndexBuffer::new(
            &base_vulkan_state.instance,
            &base_vulkan_state.physical_device,
            &base_vulkan_state.device,
            &command_pool,
            base_vulkan_state.graphics_queue,
            indices,
        );

        let uniform_buffers = base_vulkan_state.create_uniform_buffers();

        let descriptor_pool = base_vulkan_state
            .create_descriptor_pool()
            .expect("failed to create descriptor pool!");

        let descriptor_sets = base_vulkan_state
            .create_descriptor_sets(descriptor_pool, &descriptor_set_layouts)
            .expect("failed to allocate descriptor sets!");

        descriptor_sets.iter().zip(&uniform_buffers).for_each(
            |(descriptor_set, uniform_buffer)| {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffer.buffer.buffer)
                    .offset(0)
                    .range(std::mem::size_of::<UniformBufferObject>() as u64);

                let buffer_infos = [buffer_info];
                let uniform_write_descriptor_set = vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&buffer_infos);

                let image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture_image_view)
                    .sampler(texture_sampler);
                let image_infos = [image_info];
                let sampler_write_descriptor_set = vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&image_infos);

                let descriptor_writes =
                    [uniform_write_descriptor_set, sampler_write_descriptor_set];
                let descriptor_copies = [];
                unsafe {
                    base_vulkan_state
                        .device
                        .update_descriptor_sets(&descriptor_writes, &descriptor_copies)
                };
            },
        );

        let command_buffers = base_vulkan_state
            .create_command_buffers(&command_pool)
            .expect("failed to allocate command buffers!");

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&base_vulkan_state.device);

        Self {
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
            framebuffer_resized: false,
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_sets,
            start_instant: Instant::now(),
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            base_vulkan_state,
            swapchain,
            swapchain_image_views,
            swapchain_images,
            swapchain_loader,
            extent,
            format,
            depth_image,
            depth_image_memory,
            depth_image_view,
        }
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
                unsafe { self.base_vulkan_state.device.device_wait_idle() }
                    .expect("failed to wait for idle on exit!");
                elwt.exit()
            }
            Event::AboutToWait => {
                //AboutToWait is the new MainEventsCleared
                self.base_vulkan_state.window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id: _,
            } => {
                let size = self.base_vulkan_state.window.inner_size();
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
            self.base_vulkan_state.device.wait_for_fences(
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
            self.base_vulkan_state
                .device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
        }
        .expect("failed to reset in flight fence!");

        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.base_vulkan_state.device.reset_command_buffer(
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
            self.base_vulkan_state.device.queue_submit(
                self.base_vulkan_state.graphics_queue,
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
                .queue_present(self.base_vulkan_state.present_queue, &present_info)
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
        let now = Instant::now();
        let dur = now - self.start_instant;
        let time = dur.as_millis();
        let model_angle = (time as f32 / 100000.) * 90.;
        let mut uniform_buffer_object = UniformBufferObject {
            model: glam::Mat4::IDENTITY * glam::Mat4::from_axis_angle(glam::Vec3::Z, model_angle),
            view: glam::Mat4::look_at_rh(
                glam::Vec3::new(2., 2., 2.),
                glam::Vec3::new(0., 0., 0.),
                glam::Vec3::Z,
            ),
            proj: glam::Mat4::perspective_rh_gl(
                45.,
                self.extent.width as f32 / self.extent.height as f32,
                0.1,
                10.,
            ),
        };
        uniform_buffer_object.proj.y_axis *= -1.;

        let uniform_buffer = &mut self.uniform_buffers[current_image];
        uniform_buffer.modify_buffer(vec![uniform_buffer_object]);
    }

    fn recreate_swapchain(&mut self) {
        //may want to look into using the old swapchain to create a new one instead of just waiting for idle then destroying the current one.
        unsafe { self.base_vulkan_state.device.device_wait_idle() }
            .expect("failed to wait for device idle!");

        self.cleanup_swapchain();

        let swapchain_details = SwapchainSupportDetails::new(
            &self.base_vulkan_state.physical_device,
            &self.base_vulkan_state.surface_loader,
            &self.base_vulkan_state.surface,
        );

        self.base_vulkan_state.swapchain_support = swapchain_details;

        let window_size = self.base_vulkan_state.window.inner_size();

        let (swapchain, swapchain_loader, format, extent) = self
            .base_vulkan_state
            .create_swapchain(window_size.width, window_size.height)
            .expect("failed to recreate swapchain!");
        self.swapchain = swapchain;
        self.swapchain_loader = swapchain_loader;
        self.format = format;
        self.extent = extent;

        self.swapchain_images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .expect("failed to get swapchain images!");

        self.swapchain_image_views = self
            .base_vulkan_state
            .create_swapchain_image_views(&self.swapchain_images, format);

        let (depth_image, depth_image_memory, depth_image_view) =
            self.base_vulkan_state.create_depth_resources(extent);

        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;

        self.swapchain_framebuffers = Self::create_frame_buffers(
            &self.swapchain_image_views,
            &self.render_pass,
            &self.extent,
            &self.base_vulkan_state.device,
            self.depth_image_view,
        );
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

    fn create_sub_pass<'a>(
        color_attachment_refs: &'a Vec<vk::AttachmentReference>,
        depth_stencil_attachment: &'a vk::AttachmentReference,
        pipeline_bind_point: vk::PipelineBindPoint,
    ) -> vk::SubpassDescription<'a> {
        let subpass = vk::SubpassDescription::default()
            .color_attachments(color_attachment_refs)
            .depth_stencil_attachment(depth_stencil_attachment)
            .pipeline_bind_point(pipeline_bind_point);
        subpass
    }

    fn create_frame_buffers(
        swapchain_image_views: &Vec<ImageView>,
        render_pass: &vk::RenderPass,
        swapchain_extent: &Extent2D,
        device: &Device,
        depth_image_view: vk::ImageView,
    ) -> Vec<vk::Framebuffer> {
        swapchain_image_views
            .iter()
            .map(|swapchain_image_view| {
                let attachments = [*swapchain_image_view, depth_image_view];
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

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.base_vulkan_state
                .device
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .expect("failed to begin recording command buffer!");

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue::default().depth(1.).stencil(0),
            },
        ];
        let extent = self.extent;

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.base_vulkan_state.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            )
        };

        unsafe {
            self.base_vulkan_state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.)
            .max_depth(1.);
        unsafe {
            self.base_vulkan_state
                .device
                .cmd_set_viewport(command_buffer, 0, &[viewport])
        };

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);
        unsafe {
            self.base_vulkan_state
                .device
                .cmd_set_scissor(command_buffer, 0, &[scissor])
        };

        let vertex_buffers = [self.vertex_buffer.buffer.buffer];
        let offsets = [0];
        unsafe {
            self.base_vulkan_state.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &vertex_buffers,
                &offsets,
            )
        }
        unsafe {
            self.base_vulkan_state.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer.buffer,
                0,
                vk::IndexType::UINT32,
            )
        };
        unsafe {
            self.base_vulkan_state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            )
        };
        unsafe {
            self.base_vulkan_state.device.cmd_draw_indexed(
                command_buffer,
                self.index_buffer.index_count as u32,
                1,
                0,
                0,
                0,
            )
        };

        unsafe {
            self.base_vulkan_state
                .device
                .cmd_end_render_pass(command_buffer)
        };

        unsafe {
            self.base_vulkan_state
                .device
                .end_command_buffer(command_buffer)
        }
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
            self.base_vulkan_state
                .device
                .destroy_image_view(self.depth_image_view, None);
            self.base_vulkan_state
                .device
                .destroy_image(self.depth_image, None);
            self.base_vulkan_state
                .device
                .free_memory(self.depth_image_memory, None);
            self.swapchain_framebuffers.iter().for_each(|framebuffer| {
                self.base_vulkan_state
                    .device
                    .destroy_framebuffer(*framebuffer, None)
            });
            self.swapchain_image_views.iter().for_each(|image_view| {
                self.base_vulkan_state
                    .device
                    .destroy_image_view(*image_view, None)
            });
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAXFRAMESINFLIGHT {
                self.base_vulkan_state
                    .device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.base_vulkan_state
                    .device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.base_vulkan_state
                    .device
                    .destroy_fence(self.in_flight_fences[i], None);
            }
            self.base_vulkan_state
                .device
                .destroy_command_pool(self.command_pool, None);
            self.cleanup_swapchain();
            self.base_vulkan_state
                .device
                .destroy_sampler(self.texture_sampler, None);
            self.base_vulkan_state
                .device
                .destroy_image_view(self.texture_image_view, None);
            self.base_vulkan_state
                .device
                .destroy_image(self.texture_image, None);
            self.base_vulkan_state
                .device
                .free_memory(self.texture_image_memory, None);
            for i in 0..MAXFRAMESINFLIGHT {
                let uniform_buffer = &mut self.uniform_buffers[i];
                uniform_buffer.cleanup(&self.base_vulkan_state.device);
            }
            self.base_vulkan_state
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for i in 0..self.descriptor_set_layouts.len() {
                let descriptor_set_layout = self.descriptor_set_layouts[i];
                self.base_vulkan_state
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            self.index_buffer.cleanup(&self.base_vulkan_state.device);
            self.vertex_buffer.cleanup(&self.base_vulkan_state.device);
            self.base_vulkan_state
                .device
                .destroy_pipeline(self.graphics_pipeline, None);
            self.base_vulkan_state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.base_vulkan_state
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

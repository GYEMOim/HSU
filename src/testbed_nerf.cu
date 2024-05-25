/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 /** @file   testbed_nerf.cu
  *  @author Thomas Müller & Alex Evans, NVIDIA
  */

#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
  //#include <neural-graphics-primitives/device_launch_parameters.h>
  //#include <neural-graphics-primitives/mat4_sm.h>
#include <gl/glut.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/freeglut_ext.h>
#include <gl/freeglut.h>
#include <gl/freeglut_std.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/spherical_harmonics.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>
#include <imgui/imgui.h>
#include <GLFW/glfw3.h>

#ifdef copysign
#undef copysign
#endif

namespace ngp {

	static constexpr uint32_t MARCH_ITER = 10000;

	static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
	static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

	float* crossProduct(float p1[3], float p2[3]) {
		float result[3];

		result[0] = (p1[1] * p2[2]) - (p1[2] * p2[1]);
		result[1] = (p1[2] * p2[0]) - (p1[0] * p2[2]);
		result[2] = (p1[0] * p2[1]) - (p1[1] * p2[0]);

		return result;
	}

	float* makeVector(const float p1[3], const float p2[3]) //from two vertices make vector
	{
		float vector[3];
		vector[0] = p2[0] - p1[0];
		vector[1] = p2[1] - p1[1];
		vector[2] = p2[2] - p1[2];
		return vector;
	}

	float* normalizeF(float v[3]) {
		float vector[3];
		float mag = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (mag > 0) {
			vector[0] = v[0] / mag;
			vector[1] = v[1] / mag;
			vector[2] = v[2] / mag;

		}
		return  vector;
	}

	__host__ __device__ void crossProduct(const float p1[3], const float p2[3], float result[3]) {
		result[0] = (p1[1] * p2[2]) - (p1[2] * p2[1]);
		result[1] = (p1[2] * p2[0]) - (p1[0] * p2[2]);
		result[2] = (p1[0] * p2[1]) - (p1[1] * p2[0]);
	}

	__host__ __device__ float dotProduct(const float p1[3], const float p2[3]) {
		return (p1[0] * p2[0]) + (p1[1] * p2[1]) + (p1[2] * p2[2]);
	}

	__host__ __device__ void makeVector(const float p1[3], const float p2[3], float result[3]) {
		result[0] = p2[0] - p1[0];
		result[1] = p2[1] - p1[1];
		result[2] = p2[2] - p1[2];
	}

	__host__ __device__ static float GetLength(float pos1[3], float pos2[3]) {
		return sqrtf((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]) + (pos1[2] - pos2[2]) * (pos1[2] - pos2[2]));
	}

	__host__ __device__ float squaredNorm(const float* vector) {
		return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
	}

	__device__ void normalize(float v[3]) {
		float mag = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (mag > 0) {
			v[0] /= mag;
			v[1] /= mag;
			v[2] /= mag;
		}
	}

	Testbed::NetworkDims Testbed::network_dims_nerf() const {
		NetworkDims dims;
		dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
		dims.n_output = 4;
		dims.n_pos = sizeof(NerfPosition) / sizeof(float);
		return dims;
	}

	__global__ void extract_srgb_with_activation(const uint32_t n_elements, const uint32_t rgb_stride, const float* __restrict__ rgbd, float* __restrict__ rgb, ENerfActivation rgb_activation, bool from_linear) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		const uint32_t elem_idx = i / 3;
		const uint32_t dim_idx = i - elem_idx * 3;

		float c = network_to_rgb(rgbd[elem_idx * 4 + dim_idx], rgb_activation);
		if (from_linear) {
			c = linear_to_srgb(c);
		}

		rgb[elem_idx * rgb_stride + dim_idx] = c;
	}

	__global__ void mark_untrained_density_grid(const uint32_t n_elements, float* __restrict__ grid_out,
		const uint32_t n_training_images,
		const TrainingImageMetadata* __restrict__ metadata,
		const TrainingXForm* training_xforms,
		bool clear_visible_voxels
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		uint32_t level = i / NERF_GRID_N_CELLS();
		uint32_t pos_idx = i % NERF_GRID_N_CELLS();

		uint32_t x = morton3D_invert(pos_idx >> 0);
		uint32_t y = morton3D_invert(pos_idx >> 1);
		uint32_t z = morton3D_invert(pos_idx >> 2);

		float voxel_size = scalbnf(1.0f / NERF_GRIDSIZE(), level);
		vec3 pos = (vec3{ (float)x, (float)y, (float)z } / (float)NERF_GRIDSIZE() - 0.5f) * scalbnf(1.0f, level) + 0.5f;

		vec3 corners[8] = {
			pos + vec3{0.0f,       0.0f,       0.0f      },
			pos + vec3{voxel_size, 0.0f,       0.0f      },
			pos + vec3{0.0f,       voxel_size, 0.0f      },
			pos + vec3{voxel_size, voxel_size, 0.0f      },
			pos + vec3{0.0f,       0.0f,       voxel_size},
			pos + vec3{voxel_size, 0.0f,       voxel_size},
			pos + vec3{0.0f,       voxel_size, voxel_size},
			pos + vec3{voxel_size, voxel_size, voxel_size},
		};

		// Number of training views that need to see a voxel cell
		// at minimum for that cell to be marked trainable.
		// Floaters can be reduced by increasing this value to 2,
		// but at the cost of certain reconstruction artifacts.
		const uint32_t min_count = 1;
		uint32_t count = 0;

		for (uint32_t j = 0; j < n_training_images && count < min_count; ++j) {
			const auto& xform = training_xforms[j].start;
			const auto& m = metadata[j];

			if (m.lens.mode == ELensMode::FTheta || m.lens.mode == ELensMode::LatLong || m.lens.mode == ELensMode::Equirectangular) {
				// FTheta lenses don't have a forward mapping, so are assumed seeing everything. Latlong and equirect lenses
				// by definition see everything.
				++count;
				continue;
			}

			for (uint32_t k = 0; k < 8; ++k) {
				// Only consider voxel corners in front of the camera
				vec3 dir = normalize(corners[k] - xform[3]);
				if (dot(dir, xform[2]) < 1e-4f) {
					continue;
				}

				// Check if voxel corner projects onto the image plane, i.e. uv must be in (0, 1)^2
				vec2 uv = pos_to_uv(corners[k], m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), {}, m.lens);

				// `pos_to_uv` is _not_ injective in the presence of lens distortion (which breaks down outside of the image plane).
				// So we need to check whether the produced uv location generates a ray that matches the ray that we started with.
				Ray ray = uv_to_ray(0.0f, uv, m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, m.lens);
				if (distance(normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
					++count;
					break;
				}
			}
		}

		if (clear_visible_voxels || (grid_out[i] < 0) != (count < min_count)) {
			grid_out[i] = (count >= min_count) ? 0.f : -1.f;
		}
	}

	__global__ void generate_grid_samples_nerf_uniform(ivec3 res_3d, const uint32_t step, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb, NerfPosition* __restrict__ out) {
		// check grid_in for negative values -> must be negative on output
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
		uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
		if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
			return;
		}

		uint32_t i = x + y * res_3d.x + z * res_3d.x * res_3d.y;
		vec3 pos = vec3{ (float)x, (float)y, (float)z } / vec3(res_3d - 1);
		pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);
		out[i] = { warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE()) };
	}

	// generate samples for uniform grid including constant ray direction
	__global__ void generate_grid_samples_nerf_uniform_dir(ivec3 res_3d, const uint32_t step, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb, vec3 ray_dir, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims, bool voxel_centers) {
		// check grid_in for negative values -> must be negative on output
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
		uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
		if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
			return;
		}

		uint32_t i = x + y * res_3d.x + z * res_3d.x * res_3d.y;
		vec3 pos;
		if (voxel_centers) {
			pos = vec3{ (float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f } / vec3(res_3d);
		}
		else {
			pos = vec3{ (float)x, (float)y, (float)z } / vec3(res_3d - 1);
		}

		pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);

		network_input(i)->set_with_optional_extra_dims(warp_position(pos, train_aabb), warp_direction(ray_dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
	}

	__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		// 1 random number to select the level, 3 to select the position.
		rng.advance(i * 4);
		uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

		// Select grid cell that has density
		uint32_t idx;
		for (uint32_t j = 0; j < 10; ++j) {
			idx = ((i + step * n_elements) * 56924617 + j * 19349663 + 96925573) % NERF_GRID_N_CELLS();
			idx += level * NERF_GRID_N_CELLS();
			if (grid_in[idx] > thresh) {
				break;
			}
		}

		// Random position within that cellq
		uint32_t pos_idx = idx % NERF_GRID_N_CELLS();

		uint32_t x = morton3D_invert(pos_idx >> 0);
		uint32_t y = morton3D_invert(pos_idx >> 1);
		uint32_t z = morton3D_invert(pos_idx >> 2);

		vec3 pos = ((vec3{ (float)x, (float)y, (float)z } + random_val_3d(rng)) / (float)NERF_GRIDSIZE() - 0.5f) * scalbnf(1.0f, level) + 0.5f;

		out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
		indices[i] = idx;
	}

	__global__ void splat_grid_samples_nerf_max_nearest_neighbor(const uint32_t n_elements, const uint32_t* __restrict__ indices, const network_precision_t* network_output, float* __restrict__ grid_out, ENerfActivation rgb_activation, ENerfActivation density_activation) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		uint32_t local_idx = indices[i];

		// Current setting: optical thickness of the smallest possible stepsize.
		// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
		uint32_t level = 0;//local_idx / NERF_GRID_N_CELLS();

		float mlp = network_to_density(float(network_output[i]), density_activation);
		float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

		// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
		// uint atomicMax is thus perfectly acceptable.
		atomicMax((uint32_t*)&grid_out[local_idx], __float_as_uint(optical_thickness));
	}

	__global__ void grid_samples_half_to_float(const uint32_t n_elements, BoundingBox aabb, float* dst, const network_precision_t* network_output, ENerfActivation density_activation, const NerfPosition* __restrict__ coords_in, const float* __restrict__ grid_in, uint32_t max_cascade) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		// let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
		//float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
		float mlp = float(network_output[i]);

		if (grid_in) {
			vec3 pos = unwarp_position(coords_in[i].p, aabb);
			float grid_density = cascaded_grid_at(pos, grid_in, mip_from_pos(pos, max_cascade));
			if (grid_density < NERF_MIN_OPTICAL_THICKNESS()) {
				mlp = -10000.0f;
			}
		}

		dst[i] = mlp;
	}

	__global__ void ema_grid_samples_nerf(const uint32_t n_elements,
		float decay,
		const uint32_t count,
		float* __restrict__ grid_out,
		const float* __restrict__ grid_in
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		float importance = grid_in[i];

		// float ema_debias_old = 1 - (float)powf(decay, count);
		// float ema_debias_new = 1 - (float)powf(decay, count+1);

		// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
		// grid_out[i] = filtered_val;

		// Maximum instead of EMA allows capture of very thin features.
		// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

		float prev_val = grid_out[i];
		float val = (prev_val < 0.f) ? prev_val : fmaxf(prev_val * decay, importance);
		grid_out[i] = val;
	}

	__global__ void decay_sharpness_grid_nerf(const uint32_t n_elements, float decay, float* __restrict__ grid) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;
		grid[i] *= decay;
	}

	__global__ void grid_to_bitfield(
		const uint32_t n_elements,
		const uint32_t n_nonzero_elements,
		const float* __restrict__ grid,
		uint8_t* __restrict__ grid_bitfield,
		const float* __restrict__ mean_density_ptr
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;
		if (i >= n_nonzero_elements) {
			grid_bitfield[i] = 0;
			return;
		}

		uint8_t bits = 0;

		float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

		NGP_PRAGMA_UNROLL
			for (uint8_t j = 0; j < 8; ++j) {
				bits |= grid[i * 8 + j] > thresh ? ((uint8_t)1 << j) : 0;
			}

		grid_bitfield[i] = bits;
	}

	__global__ void bitfield_max_pool(const uint32_t n_elements,
		const uint8_t* __restrict__ prev_level,
		uint8_t* __restrict__ next_level
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		uint8_t bits = 0;

		NGP_PRAGMA_UNROLL
			for (uint8_t j = 0; j < 8; ++j) {
				// If any bit is set in the previous level, set this
				// level's bit. (Max pooling.)
				bits |= prev_level[i * 8 + j] > 0 ? ((uint8_t)1 << j) : 0;
			}

		uint32_t x = morton3D_invert(i >> 0) + NERF_GRIDSIZE() / 8;
		uint32_t y = morton3D_invert(i >> 1) + NERF_GRIDSIZE() / 8;
		uint32_t z = morton3D_invert(i >> 2) + NERF_GRIDSIZE() / 8;

		next_level[morton3D(x, y, z)] |= bits;
	}

	__device__ void advance_pos_nerf(
		NerfPayload& payload,
		const BoundingBox& render_aabb,
		const mat3& render_aabb_to_local,
		const vec3& camera_fwd,
		const vec2& focal_length,
		uint32_t sample_index,
		const uint8_t* __restrict__ density_grid,
		uint32_t min_mip,
		uint32_t max_mip,
		float cone_angle_constant
	) {
		if (!payload.alive) {
			return;
		}

		vec3 origin = payload.origin;
		vec3 dir = payload.dir;
		vec3 idir = vec3(1.0f) / dir;

		float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

		float t = advance_n_steps(payload.t, cone_angle, ld_random_val(sample_index, payload.idx * 786433));
		t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, { origin, dir }, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		if (t >= MAX_DEPTH()) {
			payload.alive = false;
		}
		else {
			payload.t = t;
		}
	}

	__global__ void advance_pos_nerf_kernel(
		const uint32_t n_elements,
		BoundingBox render_aabb,
		mat3 render_aabb_to_local,
		vec3 camera_fwd,
		vec2 focal_length,
		uint32_t sample_index,
		NerfPayload* __restrict__ payloads,
		const uint8_t* __restrict__ density_grid,
		uint32_t min_mip,
		uint32_t max_mip,
		float cone_angle_constant
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		advance_pos_nerf(payloads[i], render_aabb, render_aabb_to_local, camera_fwd, focal_length, sample_index, density_grid, min_mip, max_mip, cone_angle_constant);
	}

	__global__ void generate_nerf_network_inputs_from_positions(const uint32_t n_elements, BoundingBox aabb, const vec3* __restrict__ pos, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		vec3 dir = normalize(pos[i] - 0.5f); // choose outward pointing directions, for want of a better choice
		network_input(i)->set_with_optional_extra_dims(warp_position(pos[i], aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
	}

	__global__ void generate_nerf_network_inputs_at_current_position(const uint32_t n_elements, BoundingBox aabb, const NerfPayload* __restrict__ payloads, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		vec3 dir = payloads[i].dir;
		network_input(i)->set_with_optional_extra_dims(warp_position(payloads[i].origin + dir * payloads[i].t, aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
	}

	__device__ vec4 compute_nerf_rgba(const vec4& network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
		vec4 rgba = network_output;

		float density = network_to_density(rgba.a, density_activation);
		float alpha = 1.f;
		if (density_as_alpha) {
			rgba.a = density;
		}
		else {
			rgba.a = alpha = clamp(1.f - __expf(-density * depth), 0.0f, 1.0f);
		}

		rgba.rgb() = network_to_rgb_vec(rgba.rgb(), rgb_activation) * alpha;
		return rgba;
	}

	__global__ void compute_nerf_rgba_kernel(const uint32_t n_elements, vec4* network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		/*int volume_dim = 256;
		int margin = 50;
		int ix = i % volume_dim;
		int iy = (i / volume_dim) % volume_dim;
		int iz = i / (volume_dim * volume_dim);*/

		if (i >= n_elements) return;
		//|| ix < margin || ix >= volume_dim - margin || iy < margin || iy >= volume_dim - margin || iz < margin || iz >= volume_dim - margin
		network_output[i] = compute_nerf_rgba(network_output[i], rgb_activation, density_activation, depth, density_as_alpha);
		if (network_output[i].w < 5.0f) {
			network_output[i].w = 0.0f;
		}
		network_output[i].w = 1.0f - pow(2.71828, -(network_output[i].w));

	}

	__global__ void generate_next_nerf_network_inputs(
		const uint32_t n_elements,
		BoundingBox render_aabb,
		mat3 render_aabb_to_local,
		BoundingBox train_aabb,
		vec2 focal_length,
		vec3 camera_fwd,
		NerfPayload* __restrict__ payloads,
		PitchedPtr<NerfCoordinate> network_input,
		uint32_t n_steps,
		const uint8_t* __restrict__ density_grid,
		uint32_t min_mip,
		uint32_t max_mip,
		float cone_angle_constant,
		const float* extra_dims
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		NerfPayload& payload = payloads[i];

		if (!payload.alive) {
			return;
		}

		vec3 origin = payload.origin;
		vec3 dir = payload.dir;
		vec3 idir = vec3(1.0f) / dir;

		float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

		float t = payload.t;

		for (uint32_t j = 0; j < n_steps; ++j) {
			t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, { origin, dir }, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
			if (t >= MAX_DEPTH()) {
				payload.n_steps = j;
				return;
			}

			float dt = calc_dt(t, cone_angle);
			network_input(i + j * n_elements)->set_with_optional_extra_dims(warp_position(origin + dir * t, train_aabb), warp_direction(dir), warp_dt(dt), extra_dims, network_input.stride_in_bytes); // XXXCONE
			t += dt;
		}

		payload.t = t;
		payload.n_steps = n_steps;
	}

	__global__ void composite_kernel_nerf(
		const uint32_t n_elements,
		const uint32_t stride,
		const uint32_t current_step,
		BoundingBox aabb,
		float glow_y_cutoff,
		int glow_mode,
		mat4x3 camera_matrix,
		vec2 focal_length,
		float depth_scale,
		vec4* __restrict__ rgba,
		float* __restrict__ depth,
		NerfPayload* payloads,
		PitchedPtr<NerfCoordinate> network_input,
		const network_precision_t* __restrict__ network_output,
		uint32_t padded_output_width,
		uint32_t n_steps,
		ERenderMode render_mode,
		const uint8_t* __restrict__ density_grid,
		ENerfActivation rgb_activation,
		ENerfActivation density_activation,
		int show_accel,
		float min_transmittance
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		NerfPayload& payload = payloads[i];

		if (!payload.alive) {
			return;
		}

		vec4 local_rgba = rgba[i];
		float local_depth = depth[i];
		vec3 origin = payload.origin;
		vec3 cam_fwd = camera_matrix[2];
		// Composite in the last n steps
		uint32_t actual_n_steps = payload.n_steps;
		uint32_t j = 0;

		for (; j < actual_n_steps; ++j) {
			tvec<network_precision_t, 4> local_network_output;
			local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
			local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
			local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
			local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
			const NerfCoordinate* input = network_input(i + j * n_elements);
			vec3 warped_pos = input->pos.p;
			vec3 pos = unwarp_position(warped_pos, aabb);

			float T = 1.f - local_rgba.a;
			float dt = unwarp_dt(input->dt);
			float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
			if (show_accel >= 0) {
				alpha = 1.f;
			}
			float weight = alpha * T;

			vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

			if (glow_mode) { // random grid visualizations ftw!
#if 0
				if (0) {  // extremely startrek edition
					float glow_y = (pos.y - (glow_y_cutoff - 0.5f)) * 2.f;
					if (glow_y > 1.f) glow_y = max(0.f, 21.f - glow_y * 20.f);
					if (glow_y > 0.f) {
						float line;
						line = max(0.f, cosf(pos.y * 2.f * 3.141592653589793f * 16.f) - 0.95f);
						line += max(0.f, cosf(pos.x * 2.f * 3.141592653589793f * 16.f) - 0.95f);
						line += max(0.f, cosf(pos.z * 2.f * 3.141592653589793f * 16.f) - 0.95f);
						line += max(0.f, cosf(pos.y * 4.f * 3.141592653589793f * 16.f) - 0.975f);
						line += max(0.f, cosf(pos.x * 4.f * 3.141592653589793f * 16.f) - 0.975f);
						line += max(0.f, cosf(pos.z * 4.f * 3.141592653589793f * 16.f) - 0.975f);
						glow_y = glow_y * glow_y * 0.5f + glow_y * line * 25.f;
						rgb.y += glow_y;
						rgb.z += glow_y * 0.5f;
						rgb.x += glow_y * 0.25f;
					}
				}
#endif
				float glow = 0.f;

				bool green_grid = glow_mode & 1;
				bool green_cutline = glow_mode & 2;
				bool mask_to_alpha = glow_mode & 4;

				// less used?
				bool radial_mode = glow_mode & 8;
				bool grid_mode = glow_mode & 16; // makes object rgb go black!

				{
					float dist;
					if (radial_mode) {
						dist = distance(pos, camera_matrix[3]);
						dist = min(dist, (4.5f - pos.y) * 0.333f);
					}
					else {
						dist = pos.y;
					}

					if (grid_mode) {
						glow = 1.f / max(1.f, dist);
					}
					else {
						float y = glow_y_cutoff - dist; // - (ii*0.005f);
						float mask = 0.f;
						if (y > 0.f) {
							y *= 80.f;
							mask = min(1.f, y);
							//if (mask_mode) {
							//	rgb.x=rgb.y=rgb.z=mask; // mask mode
							//} else
							{
								if (green_cutline) {
									glow += max(0.f, 1.f - abs(1.f - y)) * 4.f;
								}

								if (y > 1.f) {
									y = 1.f - (y - 1.f) * 0.05f;
								}

								if (green_grid) {
									glow += max(0.f, y / max(1.f, dist));
								}
							}
						}
						if (mask_to_alpha) {
							weight *= mask;
						}
					}
				}

				if (glow > 0.f) {
					float line;
					line = max(0.f, cosf(pos.y * 2.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.x * 2.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.z * 2.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.y * 4.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.x * 4.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.z * 4.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.y * 8.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.x * 8.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.z * 8.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.y * 16.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.x * 16.f * 3.141592653589793f * 16.f) - 0.975f);
					line += max(0.f, cosf(pos.z * 16.f * 3.141592653589793f * 16.f) - 0.975f);
					if (grid_mode) {
						glow = /*glow*glow*0.75f + */ glow * line * 15.f;
						rgb.y = glow;
						rgb.z = glow * 0.5f;
						rgb.x = glow * 0.25f;
					}
					else {
						glow = glow * glow * 0.25f + glow * line * 15.f;
						rgb.y += glow;
						rgb.z += glow * 0.5f;
						rgb.x += glow * 0.25f;
					}
				}
			} // glow

			if (render_mode == ERenderMode::Normals) {
				// Network input contains the gradient of the network output w.r.t. input.
				// So to compute density gradients, we need to apply the chain rule.
				// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
				vec3 normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
				rgb = normalize(normal);
			}
			else if (render_mode == ERenderMode::Positions) {
				rgb = (pos - 0.5f) / 2.0f + 0.5f;
			}
			else if (render_mode == ERenderMode::EncodingVis) {
				rgb = warped_pos;
			}
			else if (render_mode == ERenderMode::Depth) {
				rgb = vec3(dot(cam_fwd, pos - origin) * depth_scale);
			}
			else if (render_mode == ERenderMode::AO) {
				rgb = vec3(alpha);
			}

			if (show_accel >= 0) {
				uint32_t mip = max((uint32_t)show_accel, mip_from_pos(pos));
				uint32_t res = NERF_GRIDSIZE() >> mip;
				int ix = pos.x * res;
				int iy = pos.y * res;
				int iz = pos.z * res;
				default_rng_t rng(ix + iy * 232323 + iz * 727272);
				rgb.x = 1.f - mip * (1.f / (NERF_CASCADES() - 1));
				rgb.y = rng.next_float();
				rgb.z = rng.next_float();
			}

			local_rgba += vec4(rgb * weight, weight);
			if (weight > payload.max_weight) {
				payload.max_weight = weight;
				local_depth = dot(cam_fwd, pos - camera_matrix[3]);
			}

			if (local_rgba.a > (1.0f - min_transmittance)) {
				local_rgba /= local_rgba.a;
				break;
			}
		}

		if (j < n_steps) {
			payload.alive = false;
			payload.n_steps = j + current_step;
		}

		rgba[i] = local_rgba;
		depth[i] = local_depth;
	}

	__global__ void generate_training_samples_nerf(
		const uint32_t n_rays,
		BoundingBox aabb,
		const uint32_t max_samples,
		const uint32_t n_rays_total,
		default_rng_t rng,
		uint32_t* __restrict__ ray_counter,
		uint32_t* __restrict__ numsteps_counter,
		uint32_t* __restrict__ ray_indices_out,
		Ray* __restrict__ rays_out_unnormalized,
		uint32_t* __restrict__ numsteps_out,
		PitchedPtr<NerfCoordinate> coords_out,
		const uint32_t n_training_images,
		const TrainingImageMetadata* __restrict__ metadata,
		const TrainingXForm* training_xforms,
		const uint8_t* __restrict__ density_grid,
		uint32_t max_mip,
		bool max_level_rand_training,
		float* __restrict__ max_level_ptr,
		bool snap_to_pixel_centers,
		bool train_envmap,
		float cone_angle_constant,
		Buffer2DView<const vec2> distortion,
		const float* __restrict__ cdf_x_cond_y,
		const float* __restrict__ cdf_y,
		const float* __restrict__ cdf_img,
		const ivec2 cdf_res,
		const float* __restrict__ extra_dims_gpu,
		uint32_t n_extra_dims
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_rays) return;

		uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
		ivec2 resolution = metadata[img].resolution;

		rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
		vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

		// Negative values indicate masked-away regions
		size_t pix_idx = pixel_idx(uv, resolution, 0);
		if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
			return;
		}

		float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

		float motionblur_time = random_val(rng);

		const vec2 focal_length = metadata[img].focal_length;
		const vec2 principal_point = metadata[img].principal_point;
		const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
		const Lens lens = metadata[img].lens;

		const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

		Ray ray_unnormalized;
		const Ray* rays_in_unnormalized = metadata[img].rays;
		if (rays_in_unnormalized) {
			// Rays have been explicitly supplied. Read them.
			ray_unnormalized = rays_in_unnormalized[pix_idx];

			/* DEBUG - compare the stored rays to the computed ones
			const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, 0.f);
			Ray ray2;
			ray2.o = xform[3];
			ray2.d = f_theta_distortion(uv, principal_point, lens);
			ray2.d = (xform.block<3, 3>(0, 0) * ray2.d).normalized();
			if (i==1000) {
				printf("\n%d uv %0.3f,%0.3f pixel %0.2f,%0.2f transform from [%0.5f %0.5f %0.5f] to [%0.5f %0.5f %0.5f]\n"
					" origin    [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
					" direction [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
				, img,uv.x, uv.y, uv.x*resolution.x, uv.y*resolution.y,
					training_xforms[img].start[3].x,training_xforms[img].start[3].y,training_xforms[img].start[3].z,
					training_xforms[img].end[3].x,training_xforms[img].end[3].y,training_xforms[img].end[3].z,
					ray_unnormalized.o.x,ray_unnormalized.o.y,ray_unnormalized.o.z,
					ray2.o.x,ray2.o.y,ray2.o.z,
					ray_unnormalized.d.x,ray_unnormalized.d.y,ray_unnormalized.d.z,
					ray2.d.x,ray2.d.y,ray2.d.z);
			}
			*/
		}
		else {
			ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
			if (!ray_unnormalized.is_valid()) {
				ray_unnormalized = { xform[3], xform[2] };
			}
		}

		vec3 ray_d_normalized = normalize(ray_unnormalized.d);

		vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
		float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

		// The near distance prevents learning of camera-specific fudge right in front of the camera
		tminmax.x = fmaxf(tminmax.x, 0.0f);

		float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
		vec3 idir = vec3(1.0f) / ray_d_normalized;

		// first pass to compute an accurate number of steps
		uint32_t j = 0;
		float t = startt;
		vec3 pos;

		while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
			float dt = calc_dt(t, cone_angle);
			uint32_t mip = mip_from_dt(dt, pos, max_mip);
			if (density_grid_occupied_at(pos, density_grid, mip)) {
				++j;
				t += dt;
			}
			else {
				t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
			}
		}
		if (j == 0 && !train_envmap) {
			return;
		}
		uint32_t numsteps = j;
		uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
		if (base + numsteps > max_samples) {
			return;
		}

		coords_out += base;

		uint32_t ray_idx = atomicAdd(ray_counter, 1);

		ray_indices_out[ray_idx] = i;
		rays_out_unnormalized[ray_idx] = ray_unnormalized;
		numsteps_out[ray_idx * 2 + 0] = numsteps;
		numsteps_out[ray_idx * 2 + 1] = base;

		vec3 warped_dir = warp_direction(ray_d_normalized);
		t = startt;
		j = 0;
		while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
			float dt = calc_dt(t, cone_angle);
			uint32_t mip = mip_from_dt(dt, pos, max_mip);
			if (density_grid_occupied_at(pos, density_grid, mip)) {
				coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
				++j;
				t += dt;
			}
			else {
				t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
			}
		}

		if (max_level_rand_training) {
			max_level_ptr += base;
			for (j = 0; j < numsteps; ++j) {
				max_level_ptr[j] = max_level;
			}
		}
	}


	__global__ void compute_loss_kernel_train_nerf(
		const uint32_t n_rays,
		BoundingBox aabb,
		const uint32_t n_rays_total,
		default_rng_t rng,
		const uint32_t max_samples_compacted,
		const uint32_t* __restrict__ rays_counter,
		float loss_scale,
		int padded_output_width,
		Buffer2DView<const vec4> envmap,
		float* __restrict__ envmap_gradient,
		const ivec2 envmap_resolution,
		ELossType envmap_loss_type,
		vec3 background_color,
		EColorSpace color_space,
		bool train_with_random_bg_color,
		bool train_in_linear_colors,
		const uint32_t n_training_images,
		const TrainingImageMetadata* __restrict__ metadata,
		const network_precision_t* network_output,
		uint32_t* __restrict__ numsteps_counter,
		const uint32_t* __restrict__ ray_indices_in,
		const Ray* __restrict__ rays_in_unnormalized,
		uint32_t* __restrict__ numsteps_in,
		PitchedPtr<const NerfCoordinate> coords_in,
		PitchedPtr<NerfCoordinate> coords_out,
		network_precision_t* dloss_doutput,
		ELossType loss_type,
		ELossType depth_loss_type,
		float* __restrict__ loss_output,
		bool max_level_rand_training,
		float* __restrict__ max_level_compacted_ptr,
		ENerfActivation rgb_activation,
		ENerfActivation density_activation,
		bool snap_to_pixel_centers,
		float* __restrict__ error_map,
		const float* __restrict__ cdf_x_cond_y,
		const float* __restrict__ cdf_y,
		const float* __restrict__ cdf_img,
		const ivec2 error_map_res,
		const ivec2 error_map_cdf_res,
		const float* __restrict__ sharpness_data,
		ivec2 sharpness_resolution,
		float* __restrict__ sharpness_grid,
		float* __restrict__ density_grid,
		const float* __restrict__ mean_density_ptr,
		uint32_t max_mip,
		const vec3* __restrict__ exposure,
		vec3* __restrict__ exposure_gradient,
		float depth_supervision_lambda,
		float near_distance
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= *rays_counter) { return; }

		// grab the number of samples for this ray, and the first sample
		uint32_t numsteps = numsteps_in[i * 2 + 0];
		uint32_t base = numsteps_in[i * 2 + 1];

		coords_in += base;
		network_output += base * padded_output_width;

		float T = 1.f;

		float EPSILON = 1e-4f;

		vec3 rgb_ray = vec3(0.0f);
		vec3 hitpoint = vec3(0.0f);

		float depth_ray = 0.f;
		uint32_t compacted_numsteps = 0;
		vec3 ray_o = rays_in_unnormalized[i].o;
		for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
			if (T < EPSILON) {
				break;
			}

			const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*)network_output;
			const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
			const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
			const float dt = unwarp_dt(coords_in.ptr->dt);
			float cur_depth = distance(pos, ray_o);
			float density = network_to_density(float(local_network_output[3]), density_activation);


			const float alpha = 1.f - __expf(-density * dt);
			const float weight = alpha * T;
			rgb_ray += weight * rgb;
			hitpoint += weight * pos;
			depth_ray += weight * cur_depth;
			T *= (1.f - alpha);

			network_output += padded_output_width;
			coords_in += 1;
		}
		hitpoint /= (1.0f - T);

		// Must be same seed as above to obtain the same
		// background color.
		uint32_t ray_idx = ray_indices_in[i];
		rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

		float img_pdf = 1.0f;
		uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
		ivec2 resolution = metadata[img].resolution;

		float uv_pdf = 1.0f;
		vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &uv_pdf);
		float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level
		rng.advance(1); // motionblur_time

		if (train_with_random_bg_color) {
			background_color = random_val_3d(rng);
		}
		vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

		// Composit background behind envmap
		vec4 envmap_value;
		vec3 dir;
		if (envmap) {
			dir = normalize(rays_in_unnormalized[i].d);
			envmap_value = read_envmap(envmap, dir);
			background_color = envmap_value.rgb() + background_color * (1.0f - envmap_value.a);
		}

		vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
		// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
		// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
		vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

		vec3 rgbtarget;
		if (train_in_linear_colors || color_space == EColorSpace::Linear) {
			rgbtarget = exposure_scale * texsamp.rgb() + (1.0f - texsamp.a) * background_color;

			if (!train_in_linear_colors) {
				rgbtarget = linear_to_srgb(rgbtarget);
				background_color = linear_to_srgb(background_color);
			}
		}
		else if (color_space == EColorSpace::SRGB) {
			background_color = linear_to_srgb(background_color);
			if (texsamp.a > 0) {
				rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb() / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
			}
			else {
				rgbtarget = background_color;
			}
		}

		if (compacted_numsteps == numsteps) {
			// support arbitrary background colors
			rgb_ray += T * background_color;
		}

		// Step again, this time computing loss
		network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
		coords_in -= compacted_numsteps;

		uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
		compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
		numsteps_in[i * 2 + 0] = compacted_numsteps;
		numsteps_in[i * 2 + 1] = compacted_base;
		if (compacted_numsteps == 0) {
			return;
		}

		max_level_compacted_ptr += compacted_base;
		coords_out += compacted_base;

		dloss_doutput += compacted_base * padded_output_width;

		LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
		lg.loss /= img_pdf * uv_pdf;

		float target_depth = length(rays_in_unnormalized[i].d) * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
		LossAndGradient lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
		float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;

		// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
		// Essentially: variance reduction, but otherwise the same optimization.
		// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
		// to change the weighting of the loss function. So don't divide.
		// lg.gradient /= img_pdf * uv_pdf;

		float mean_loss = mean(lg.loss);
		if (loss_output) {
			loss_output[i] = mean_loss / (float)n_rays;
		}

		if (error_map) {
			const vec2 pos = clamp(uv * vec2(error_map_res) - 0.5f, 0.0f, vec2(error_map_res) - (1.0f + 1e-4f));
			const ivec2 pos_int = pos;
			const vec2 weight = pos - vec2(pos_int);

			ivec2 idx = clamp(pos_int, 0, resolution - 2);

			auto deposit_val = [&](int x, int y, float val) {
				atomicAdd(&error_map[img * product(error_map_res) + y * error_map_res.x + x], val);
			};

			if (sharpness_data && aabb.contains(hitpoint)) {
				ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), 0, sharpness_resolution - 1);
				float sharp = sharpness_data[img * product(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

				// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
				float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
				grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

				mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
			}

			deposit_val(idx.x, idx.y, (1 - weight.x) * (1 - weight.y) * mean_loss);
			deposit_val(idx.x + 1, idx.y, weight.x * (1 - weight.y) * mean_loss);
			deposit_val(idx.x, idx.y + 1, (1 - weight.x) * weight.y * mean_loss);
			deposit_val(idx.x + 1, idx.y + 1, weight.x * weight.y * mean_loss);
		}

		loss_scale /= n_rays;

		const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
		const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

		// now do it again computing gradients
		vec3 rgb_ray2 = { 0.f,0.f,0.f };
		float depth_ray2 = 0.f;
		T = 1.f;
		for (uint32_t j = 0; j < compacted_numsteps; ++j) {
			if (max_level_rand_training) {
				max_level_compacted_ptr[j] = max_level;
			}
			// Compact network inputs
			NerfCoordinate* coord_out = coords_out(j);
			const NerfCoordinate* coord_in = coords_in(j);
			coord_out->copy(*coord_in, coords_out.stride_in_bytes);

			const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
			float depth = distance(pos, ray_o);

			float dt = unwarp_dt(coord_in->dt);
			const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*)network_output;
			const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
			const float density = network_to_density(float(local_network_output[3]), density_activation);
			const float alpha = 1.f - __expf(-density * dt);
			const float weight = alpha * T;
			rgb_ray2 += weight * rgb;
			depth_ray2 += weight * depth;
			T *= (1.f - alpha);

			// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
			const vec3 suffix = rgb_ray - rgb_ray2;
			const vec3 dloss_by_drgb = weight * lg.gradient;

			tvec<network_precision_t, 4> local_dL_doutput;

			// chain rule to go from dloss/drgb to dloss/dmlp_output
			local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
			local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
			local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

			float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
			const float depth_suffix = depth_ray - depth_ray2;
			const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);

			float dloss_by_dmlp = density_derivative * (
				dt * (dot(lg.gradient, T * rgb - suffix) + depth_supervision)
				);

			//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
			//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

			local_dL_doutput[3] =
				loss_scale * dloss_by_dmlp +
				(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
				(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
			;

			*(tvec<network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

			dloss_doutput += padded_output_width;
			network_output += padded_output_width;
		}

		if (exposure_gradient) {
			// Assume symmetric loss
			vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

			if (!train_in_linear_colors) {
				dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
			}

			// 2^exposure * log(2)
			vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
			atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
			atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
			atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
		}

		if (compacted_numsteps == numsteps && envmap_gradient) {
			vec3 loss_gradient = lg.gradient;
			if (envmap_loss_type != loss_type) {
				loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
			}

			vec3 dloss_by_dbackground = T * loss_gradient;
			if (!train_in_linear_colors) {
				dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
			}

			tvec<network_precision_t, 4> dL_denvmap;
			dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
			dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
			dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;


			float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

			// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
			dL_denvmap[3] = (network_precision_t)0;

			deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
		}
	}


	__global__ void compute_cam_gradient_train_nerf(
		const uint32_t n_rays,
		const uint32_t n_rays_total,
		default_rng_t rng,
		const BoundingBox aabb,
		const uint32_t* __restrict__ rays_counter,
		const TrainingXForm* training_xforms,
		bool snap_to_pixel_centers,
		vec3* cam_pos_gradient,
		vec3* cam_rot_gradient,
		const uint32_t n_training_images,
		const TrainingImageMetadata* __restrict__ metadata,
		const uint32_t* __restrict__ ray_indices_in,
		const Ray* __restrict__ rays_in_unnormalized,
		uint32_t* __restrict__ numsteps_in,
		PitchedPtr<NerfCoordinate> coords,
		PitchedPtr<NerfCoordinate> coords_gradient,
		float* __restrict__ distortion_gradient,
		float* __restrict__ distortion_gradient_weight,
		const ivec2 distortion_resolution,
		vec2* cam_focal_length_gradient,
		const float* __restrict__ cdf_x_cond_y,
		const float* __restrict__ cdf_y,
		const float* __restrict__ cdf_img,
		const ivec2 error_map_res
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= *rays_counter) { return; }

		// grab the number of samples for this ray, and the first sample
		uint32_t numsteps = numsteps_in[i * 2 + 0];
		if (numsteps == 0) {
			// The ray doesn't matter. So no gradient onto the camera
			return;
		}

		uint32_t base = numsteps_in[i * 2 + 1];
		coords += base;
		coords_gradient += base;

		// Must be same seed as above to obtain the same
		// background color.
		uint32_t ray_idx = ray_indices_in[i];
		uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
		ivec2 resolution = metadata[img].resolution;

		const mat4x3& xform = training_xforms[img].start;

		Ray ray = rays_in_unnormalized[i];
		ray.d = normalize(ray.d);
		Ray ray_gradient = { vec3(0.0f), vec3(0.0f) };

		// Compute ray gradient
		for (uint32_t j = 0; j < numsteps; ++j) {
			const vec3 warped_pos = coords(j)->pos.p;
			const vec3 pos_gradient = coords_gradient(j)->pos.p * warp_position_derivative(warped_pos, aabb);
			ray_gradient.o += pos_gradient;
			const vec3 pos = unwarp_position(warped_pos, aabb);

			// Scaled by t to account for the fact that further-away objects' position
			// changes more rapidly as the direction changes.
			float t = distance(pos, ray.o);
			const vec3 dir_gradient = coords_gradient(j)->dir.d * warp_direction_derivative(coords(j)->dir.d);
			ray_gradient.d += pos_gradient * t + dir_gradient;
		}

		rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
		float uv_pdf = 1.0f;

		vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &uv_pdf);

		if (distortion_gradient) {
			// Projection of the raydir gradient onto the plane normal to raydir,
			// because that's the only degree of motion that the raydir has.
			vec3 orthogonal_ray_gradient = ray_gradient.d - ray.d * dot(ray_gradient.d, ray.d);

			// Rotate ray gradient to obtain image plane gradient.
			// This has the effect of projecting the (already projected) ray gradient from the
			// tangent plane of the sphere onto the image plane (which is correct!).
			vec3 image_plane_gradient = inverse(mat3(xform)) * orthogonal_ray_gradient;

			// Splat the resulting 2D image plane gradient into the distortion params
			deposit_image_gradient(image_plane_gradient.xy() / uv_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, uv);
		}

		if (cam_pos_gradient) {
			// Atomically reduce the ray gradient into the xform gradient
			NGP_PRAGMA_UNROLL
				for (uint32_t j = 0; j < 3; ++j) {
					atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / uv_pdf);
				}
		}

		if (cam_rot_gradient) {
			// Rotation is averaged in log-space (i.e. by averaging angle-axes).
			// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
			// orthogonal, leading to the angle_axis magnitude to equal the magnitude
			// of ray_gradient.d.
			vec3 angle_axis = cross(ray.d, ray_gradient.d);

			// Atomically reduce the ray gradient into the xform gradient
			NGP_PRAGMA_UNROLL
				for (uint32_t j = 0; j < 3; ++j) {
					atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / uv_pdf);
				}
		}
	}

	__global__ void compute_extra_dims_gradient_train_nerf(
		const uint32_t n_rays,
		const uint32_t n_rays_total,
		const uint32_t* __restrict__ rays_counter,
		float* extra_dims_gradient,
		uint32_t n_extra_dims,
		const uint32_t n_training_images,
		const uint32_t* __restrict__ ray_indices_in,
		uint32_t* __restrict__ numsteps_in,
		PitchedPtr<NerfCoordinate> coords_gradient,
		const float* __restrict__ cdf_img
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= *rays_counter) { return; }

		// grab the number of samples for this ray, and the first sample
		uint32_t numsteps = numsteps_in[i * 2 + 0];
		if (numsteps == 0) {
			// The ray doesn't matter. So no gradient onto the camera
			return;
		}
		uint32_t base = numsteps_in[i * 2 + 1];
		coords_gradient += base;
		// Must be same seed as above to obtain the same
		// background color.
		uint32_t ray_idx = ray_indices_in[i];
		uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);

		extra_dims_gradient += n_extra_dims * img;

		for (uint32_t j = 0; j < numsteps; ++j) {
			const float* src = coords_gradient(j)->get_extra_dims();
			for (uint32_t k = 0; k < n_extra_dims; ++k) {
				atomicAdd(&extra_dims_gradient[k], src[k]);
			}
		}
	}

	__global__ void shade_kernel_nerf(
		const uint32_t n_elements,
		bool gbuffer_hard_edges,
		mat4x3 camera_matrix,
		float depth_scale,
		vec4* __restrict__ rgba,
		float* __restrict__ depth,
		NerfPayload* __restrict__ payloads,
		ERenderMode render_mode,
		bool train_in_linear_colors,
		vec4* __restrict__ frame_buffer,
		float* __restrict__ depth_buffer
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements || render_mode == ERenderMode::Distortion) return;
		NerfPayload& payload = payloads[i];

		vec4 tmp = rgba[i];
		if (render_mode == ERenderMode::Normals) {
			vec3 n = normalize(tmp.xyz());
			tmp.rgb() = (0.5f * n + 0.5f) * tmp.a;
		}
		else if (render_mode == ERenderMode::Cost) {
			float col = (float)payload.n_steps / 128;
			tmp = { col, col, col, 1.0f };
		}
		else if (gbuffer_hard_edges && render_mode == ERenderMode::Depth) {
			tmp.rgb() = vec3(depth[i] * depth_scale);
		}
		else if (gbuffer_hard_edges && render_mode == ERenderMode::Positions) {
			vec3 pos = camera_matrix[3] + payload.dir / dot(payload.dir, camera_matrix[2]) * depth[i];
			tmp.rgb() = (pos - 0.5f) / 2.0f + 0.5f;
		}

		if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
			// Accumulate in linear colors
			tmp.rgb() = srgb_to_linear(tmp.rgb());
		}

		frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);
		if (render_mode != ERenderMode::Slice && tmp.a > 0.2f) {
			depth_buffer[payload.idx] = depth[i];
		}
	}

	__global__ void compact_kernel_nerf(
		const uint32_t n_elements,
		vec4* src_rgba, float* src_depth, NerfPayload* src_payloads,
		vec4* dst_rgba, float* dst_depth, NerfPayload* dst_payloads,
		vec4* dst_final_rgba, float* dst_final_depth, NerfPayload* dst_final_payloads,
		uint32_t* counter, uint32_t* finalCounter
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		NerfPayload& src_payload = src_payloads[i];

		if (src_payload.alive) {
			uint32_t idx = atomicAdd(counter, 1);
			dst_payloads[idx] = src_payload;
			dst_rgba[idx] = src_rgba[i];
			dst_depth[idx] = src_depth[i];
		}
		else if (src_rgba[i].a > 0.001f) {
			uint32_t idx = atomicAdd(finalCounter, 1);
			dst_final_payloads[idx] = src_payload;
			dst_final_rgba[idx] = src_rgba[i];
			dst_final_depth[idx] = src_depth[i];
		}
	}

	__global__ void init_rays_with_payload_kernel_nerf(
		uint32_t sample_index,
		NerfPayload* __restrict__ payloads,
		ivec2 resolution,
		vec2 focal_length,
		mat4x3 camera_matrix0,
		mat4x3 camera_matrix1,
		vec4 rolling_shutter,
		vec2 screen_center,
		vec3 parallax_shift,
		bool snap_to_pixel_centers,
		BoundingBox render_aabb,
		mat3 render_aabb_to_local,
		float near_distance,
		float plane_z,
		float aperture_size,
		Foveation foveation,
		Lens lens,
		Buffer2DView<const vec4> envmap,
		vec4* __restrict__ frame_buffer,
		float* __restrict__ depth_buffer,
		Buffer2DView<const uint8_t> hidden_area_mask,
		Buffer2DView<const vec2> distortion,
		ERenderMode render_mode
	) {
		uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
		uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

		if (x >= resolution.x || y >= resolution.y) {
			return;
		}

		uint32_t idx = x + resolution.x * y;

		if (plane_z < 0) {
			aperture_size = 0.0;
		}

		vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
		vec2 uv = vec2{ (float)x + pixel_offset.x, (float)y + pixel_offset.y } / vec2(resolution);
		mat4x3 camera = get_xform_given_rolling_shutter({ camera_matrix0, camera_matrix1 }, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

		Ray ray = uv_to_ray(
			sample_index,
			uv,
			resolution,
			focal_length,
			camera,
			screen_center,
			parallax_shift,
			near_distance,
			plane_z,
			aperture_size,
			foveation,
			hidden_area_mask,
			lens,
			distortion
		);

		NerfPayload& payload = payloads[idx];
		payload.max_weight = 0.0f;

		depth_buffer[idx] = MAX_DEPTH();

		if (!ray.is_valid()) {
			payload.origin = ray.o;
			payload.alive = false;
			return;
		}

		if (plane_z < 0) {
			float n = length(ray.d);
			payload.origin = ray.o;
			payload.dir = (1.0f / n) * ray.d;
			payload.t = -plane_z * n;
			payload.idx = idx;
			payload.n_steps = 0;
			payload.alive = false;
			depth_buffer[idx] = -plane_z;
			return;
		}

		if (render_mode == ERenderMode::Distortion) {
			vec2 uv_after_distortion = pos_to_uv(ray(1.0f), resolution, focal_length, camera, screen_center, parallax_shift, foveation);

			frame_buffer[idx].rgb() = to_rgb((uv_after_distortion - uv) * 64.0f);
			frame_buffer[idx].a = 1.0f;
			depth_buffer[idx] = 1.0f;
			payload.origin = ray(MAX_DEPTH());
			payload.alive = false;
			return;
		}

		ray.d = normalize(ray.d);

		if (envmap) {
			frame_buffer[idx] = read_envmap(envmap, ray.d);
		}

		float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

		if (!render_aabb.contains(render_aabb_to_local * ray(t))) {
			payload.origin = ray.o;
			payload.alive = false;
			return;
		}

		payload.origin = ray.o;
		payload.dir = ray.d;
		payload.t = t;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = true;
	}

	static constexpr float MIN_PDF = 0.01f;

	__global__ void construct_cdf_2d(
		uint32_t n_images,
		uint32_t height,
		uint32_t width,
		const float* __restrict__ data,
		float* __restrict__ cdf_x_cond_y,
		float* __restrict__ cdf_y
	) {
		const uint32_t y = threadIdx.x + blockIdx.x * blockDim.x;
		const uint32_t img = threadIdx.y + blockIdx.y * blockDim.y;
		if (y >= height || img >= n_images) return;

		const uint32_t offset_xy = img * height * width + y * width;
		data += offset_xy;
		cdf_x_cond_y += offset_xy;

		float cum = 0;
		for (uint32_t x = 0; x < width; ++x) {
			cum += data[x] + 1e-10f;
			cdf_x_cond_y[x] = cum;
		}

		cdf_y[img * height + y] = cum;
		float norm = __frcp_rn(cum);

		for (uint32_t x = 0; x < width; ++x) {
			cdf_x_cond_y[x] = (1.0f - MIN_PDF) * cdf_x_cond_y[x] * norm + MIN_PDF * (float)(x + 1) / (float)width;
		}
	}

	__global__ void construct_cdf_1d(
		uint32_t n_images,
		uint32_t height,
		float* __restrict__ cdf_y,
		float* __restrict__ cdf_img
	) {
		const uint32_t img = threadIdx.x + blockIdx.x * blockDim.x;
		if (img >= n_images) return;

		cdf_y += img * height;

		float cum = 0;
		for (uint32_t y = 0; y < height; ++y) {
			cum += cdf_y[y];
			cdf_y[y] = cum;
		}

		cdf_img[img] = cum;

		float norm = __frcp_rn(cum);
		for (uint32_t y = 0; y < height; ++y) {
			cdf_y[y] = (1.0f - MIN_PDF) * cdf_y[y] * norm + MIN_PDF * (float)(y + 1) / (float)height;
		}
	}

	__global__ void safe_divide(const uint32_t num_elements, float* __restrict__ inout, const float* __restrict__ divisor) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= num_elements) return;

		float local_divisor = divisor[i];
		inout[i] = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
	}

	void Testbed::NerfTracer::init_rays_from_camera(
		uint32_t sample_index,
		uint32_t padded_output_width,
		uint32_t n_extra_dims,
		const ivec2& resolution,
		const vec2& focal_length,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const vec4& rolling_shutter,
		const vec2& screen_center,
		const vec3& parallax_shift,
		bool snap_to_pixel_centers,
		const BoundingBox& render_aabb,
		const mat3& render_aabb_to_local,
		float near_distance,
		float plane_z,
		float aperture_size,
		const Foveation& foveation,
		const Lens& lens,
		const Buffer2DView<const vec4>& envmap,
		const Buffer2DView<const vec2>& distortion,
		vec4* frame_buffer,
		float* depth_buffer,
		const Buffer2DView<const uint8_t>& hidden_area_mask,
		const uint8_t* grid,
		int show_accel,
		uint32_t max_mip,
		float cone_angle_constant,
		ERenderMode render_mode,
		cudaStream_t stream
	) {
		// Make sure we have enough memory reserved to render at the requested resolution
		size_t n_pixels = (size_t)resolution.x * resolution.y;
		enlarge(n_pixels, padded_output_width, n_extra_dims, stream);

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
		init_rays_with_payload_kernel_nerf << <blocks, threads, 0, stream >> > (
			sample_index,
			m_rays[0].payload,
			resolution,
			focal_length,
			camera_matrix0,
			camera_matrix1,
			rolling_shutter,
			screen_center,
			parallax_shift,
			snap_to_pixel_centers,
			render_aabb,
			render_aabb_to_local,
			near_distance,
			plane_z,
			aperture_size,
			foveation,
			lens,
			envmap,
			frame_buffer,
			depth_buffer,
			hidden_area_mask,
			distortion,
			render_mode
			);

		m_n_rays_initialized = resolution.x * resolution.y;

		CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

		linear_kernel(advance_pos_nerf_kernel, 0, stream,
			m_n_rays_initialized,
			render_aabb,
			render_aabb_to_local,
			camera_matrix1[2],
			focal_length,
			sample_index,
			m_rays[0].payload,
			grid,
			(show_accel >= 0) ? show_accel : 0,
			max_mip,
			cone_angle_constant
		);
	}

	uint32_t Testbed::NerfTracer::trace(
		const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
		const BoundingBox& render_aabb,
		const mat3& render_aabb_to_local,
		const BoundingBox& train_aabb,
		const vec2& focal_length,
		float cone_angle_constant,
		const uint8_t* grid,
		ERenderMode render_mode,
		const mat4x3& camera_matrix,
		float depth_scale,
		int visualized_layer,
		int visualized_dim,
		ENerfActivation rgb_activation,
		ENerfActivation density_activation,
		int show_accel,
		uint32_t max_mip,
		float min_transmittance,
		float glow_y_cutoff,
		int glow_mode,
		const float* extra_dims_gpu,
		cudaStream_t stream
	) {
		if (m_n_rays_initialized == 0) {
			return 0;
		}

		CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

		uint32_t n_alive = m_n_rays_initialized;
		// m_n_rays_initialized = 0;

		uint32_t i = 1;
		uint32_t double_buffer_index = 0;
		while (i < MARCH_ITER) {
			RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
			RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
			++double_buffer_index;

			// Compact rays that did not diverge yet
			{
				CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
				linear_kernel(compact_kernel_nerf, 0, stream,
					n_alive,
					rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
					rays_current.rgba, rays_current.depth, rays_current.payload,
					m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
					m_alive_counter, m_hit_counter
				);
				CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			}

			if (n_alive == 0) {
				break;
			}

			// Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
			uint32_t target_n_queries = 2 * 1024 * 1024;
			uint32_t n_steps_between_compaction = clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

			uint32_t extra_stride = network->n_extra_dims() * sizeof(float);
			PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
			linear_kernel(generate_next_nerf_network_inputs, 0, stream,
				n_alive,
				render_aabb,
				render_aabb_to_local,
				train_aabb,
				focal_length,
				camera_matrix[2],
				rays_current.payload,
				input_data,
				n_steps_between_compaction,
				grid,
				(show_accel >= 0) ? show_accel : 0,
				max_mip,
				cone_angle_constant,
				extra_dims_gpu
			);
			uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, BATCH_SIZE_GRANULARITY);
			GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
			GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network->padded_output_width(), n_elements);
			network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

			if (render_mode == ERenderMode::Normals) {
				network->input_gradient(stream, 3, positions_matrix, positions_matrix);
			}
			else if (render_mode == ERenderMode::EncodingVis) {
				network->visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
			}

			linear_kernel(composite_kernel_nerf, 0, stream,
				n_alive,
				n_elements,
				i,
				train_aabb,
				glow_y_cutoff,
				glow_mode,
				camera_matrix,
				focal_length,
				depth_scale,
				rays_current.rgba,
				rays_current.depth,
				rays_current.payload,
				input_data,
				m_network_output,
				network->padded_output_width(),
				n_steps_between_compaction,
				render_mode,
				grid,
				rgb_activation,
				density_activation,
				show_accel,
				min_transmittance
			);

			i += n_steps_between_compaction;
		}

		uint32_t n_hit;
		CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		return n_hit;
	}

	void Testbed::NerfTracer::enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
		n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
		size_t num_floats = sizeof(NerfCoordinate) / sizeof(float) + n_extra_dims;
		auto scratch = allocate_workspace_and_distribute<
			vec4, float, NerfPayload, // m_rays[0]
			vec4, float, NerfPayload, // m_rays[1]
			vec4, float, NerfPayload, // m_rays_hit

			network_precision_t,
			float,
			uint32_t,
			uint32_t
		>(
			stream, &m_scratch_alloc,
			n_elements, n_elements, n_elements,
			n_elements, n_elements, n_elements,
			n_elements, n_elements, n_elements,
			n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
			n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
			32, // 2 full cache lines to ensure no overlap
			32  // 2 full cache lines to ensure no overlap
			);

		m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
		m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
		m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

		m_network_output = std::get<9>(scratch);
		m_network_input = std::get<10>(scratch);

		m_hit_counter = std::get<11>(scratch);
		m_alive_counter = std::get<12>(scratch);
	}

	std::vector<float> Testbed::Nerf::Training::get_extra_dims_cpu(int trainview) const {
		if (dataset.n_extra_dims() == 0) {
			return {};
		}

		if (trainview < 0 || trainview >= dataset.n_images) {
			throw std::runtime_error{ "Invalid training view." };
		}

		const float* extra_dims_src = extra_dims_gpu.data() + trainview * dataset.n_extra_dims();

		std::vector<float> extra_dims_cpu(dataset.n_extra_dims());
		CUDA_CHECK_THROW(cudaMemcpy(extra_dims_cpu.data(), extra_dims_src, dataset.n_extra_dims() * sizeof(float), cudaMemcpyDeviceToHost));

		return extra_dims_cpu;
	}

	void Testbed::Nerf::Training::update_extra_dims() {
		uint32_t n_extra_dims = dataset.n_extra_dims();
		std::vector<float> extra_dims_cpu(extra_dims_gpu.size());
		for (uint32_t i = 0; i < extra_dims_opt.size(); ++i) {
			const std::vector<float>& value = extra_dims_opt[i].variable();
			for (uint32_t j = 0; j < n_extra_dims; ++j) {
				extra_dims_cpu[i * n_extra_dims + j] = value[j];
			}
		}

		CUDA_CHECK_THROW(cudaMemcpyAsync(extra_dims_gpu.data(), extra_dims_cpu.data(), extra_dims_opt.size() * n_extra_dims * sizeof(float), cudaMemcpyHostToDevice));
	}

	void Testbed::render_nerf(
		cudaStream_t stream,
		CudaDevice& device,
		const CudaRenderBufferView& render_buffer,
		const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
		const uint8_t* density_grid_bitfield,
		const vec2& focal_length,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const vec4& rolling_shutter,
		const vec2& screen_center,
		const Foveation& foveation,
		int visualized_dimension
	) {
		float plane_z = m_slice_plane_z + m_scale;
		if (m_render_mode == ERenderMode::Slice) {
			plane_z = -plane_z;
		}

		ERenderMode render_mode = visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

		const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(stream);

		NerfTracer tracer;

		// Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled.
		// (Unless we're in distortion visualization mode, in which case the distortion grid is fine to visualize.)
		auto grid_distortion =
			m_nerf.render_with_lens_distortion && (!m_dlss || m_render_mode == ERenderMode::Distortion) ?
			m_distortion.inference_view() :
			Buffer2DView<const vec2>{};

		Lens lens = m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{};

		auto resolution = render_buffer.resolution;

		tracer.init_rays_from_camera(
			render_buffer.spp,
			nerf_network->padded_output_width(),
			nerf_network->n_extra_dims(),
			render_buffer.resolution,
			focal_length,
			camera_matrix0,
			camera_matrix1,
			rolling_shutter,
			screen_center,
			m_parallax_shift,
			m_snap_to_pixel_centers,
			m_render_aabb,
			m_render_aabb_to_local,
			m_render_near_distance,
			plane_z,
			m_aperture_size,
			foveation,
			lens,
			m_envmap.inference_view(),
			grid_distortion,
			render_buffer.frame_buffer,
			render_buffer.depth_buffer,
			render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
			density_grid_bitfield,
			m_nerf.show_accel,
			m_nerf.max_cascade,
			m_nerf.cone_angle_constant,
			render_mode,
			stream
		);

		float depth_scale = 1.0f / m_nerf.training.dataset.scale;
		bool render_2d = m_render_mode == ERenderMode::Slice || m_render_mode == ERenderMode::Distortion;

		uint32_t n_hit;
		if (render_2d) {
			n_hit = tracer.n_rays_initialized();
		}
		else {
			n_hit = tracer.trace(
				nerf_network,
				m_render_aabb,
				m_render_aabb_to_local,
				m_aabb,
				focal_length,
				m_nerf.cone_angle_constant,
				density_grid_bitfield,
				render_mode,
				camera_matrix1,
				depth_scale,
				m_visualized_layer,
				visualized_dimension,
				m_nerf.rgb_activation,
				m_nerf.density_activation,
				m_nerf.show_accel,
				m_nerf.max_cascade,
				m_nerf.render_min_transmittance,
				m_nerf.glow_y_cutoff,
				m_nerf.glow_mode,
				extra_dims_gpu,
				stream
			);
		}
		RaysNerfSoa& rays_hit = render_2d ? tracer.rays_init() : tracer.rays_hit();

		if (render_2d) {
			// Store colors in the normal buffer
			uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);
			const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + nerf_network->n_extra_dims();
			const uint32_t extra_stride = nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

			GPUMatrix<float> positions_matrix{ floats_per_coord, n_elements, stream };
			GPUMatrix<float> rgbsigma_matrix{ 4, n_elements, stream };

			linear_kernel(generate_nerf_network_inputs_at_current_position, 0, stream, n_hit, m_aabb, rays_hit.payload, PitchedPtr<NerfCoordinate>((NerfCoordinate*)positions_matrix.data(), 1, 0, extra_stride), extra_dims_gpu);

			if (visualized_dimension == -1) {
				nerf_network->inference(stream, positions_matrix, rgbsigma_matrix);
				linear_kernel(compute_nerf_rgba_kernel, 0, stream, n_hit, (vec4*)rgbsigma_matrix.data(), m_nerf.rgb_activation, m_nerf.density_activation, 0.01f, false);
			}
			else {
				nerf_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, rgbsigma_matrix);
			}

			linear_kernel(shade_kernel_nerf, 0, stream,
				n_hit,
				m_nerf.render_gbuffer_hard_edges,
				camera_matrix1,
				depth_scale,
				(vec4*)rgbsigma_matrix.data(),
				nullptr,
				rays_hit.payload,
				m_render_mode,
				m_nerf.training.linear_colors,
				render_buffer.frame_buffer,
				render_buffer.depth_buffer
			);
			return;
		}

		linear_kernel(shade_kernel_nerf, 0, stream,
			n_hit,
			m_nerf.render_gbuffer_hard_edges,
			camera_matrix1,
			depth_scale,
			rays_hit.rgba,
			rays_hit.depth,
			rays_hit.payload,
			m_render_mode,
			m_nerf.training.linear_colors,
			render_buffer.frame_buffer,
			render_buffer.depth_buffer
		);

		if (render_mode == ERenderMode::Cost) {
			std::vector<NerfPayload> payloads_final_cpu(n_hit);
			CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(NerfPayload), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			size_t total_n_steps = 0;
			for (uint32_t i = 0; i < n_hit; ++i) {
				total_n_steps += payloads_final_cpu[i].n_steps;
			}
			tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps / (float)n_hit);
		}
	}

	void Testbed::Nerf::Training::set_camera_intrinsics(int frame_idx, float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2, float k3, float k4, bool is_fisheye) {
		if (frame_idx < 0 || frame_idx >= dataset.n_images) {
			return;
		}
		if (fx <= 0.f) fx = fy;
		if (fy <= 0.f) fy = fx;
		auto& m = dataset.metadata[frame_idx];
		if (cx < 0.f) cx = -cx; else cx = cx / m.resolution.x;
		if (cy < 0.f) cy = -cy; else cy = cy / m.resolution.y;
		m.lens = { ELensMode::Perspective };
		if (k1 || k2 || k3 || k4 || p1 || p2) {
			if (is_fisheye) {
				m.lens = { ELensMode::OpenCVFisheye, k1, k2, k3, k4 };
			}
			else {
				m.lens = { ELensMode::OpenCV, k1, k2, p1, p2 };
			}
		}

		m.principal_point = { cx, cy };
		m.focal_length = { fx, fy };
		dataset.update_metadata(frame_idx, frame_idx + 1);
	}

	void Testbed::Nerf::Training::set_camera_extrinsics_rolling_shutter(int frame_idx, mat4x3 camera_to_world_start, mat4x3 camera_to_world_end, const vec4& rolling_shutter, bool convert_to_ngp) {
		if (frame_idx < 0 || frame_idx >= dataset.n_images) {
			return;
		}

		if (convert_to_ngp) {
			camera_to_world_start = dataset.nerf_matrix_to_ngp(camera_to_world_start);
			camera_to_world_end = dataset.nerf_matrix_to_ngp(camera_to_world_end);
		}

		dataset.xforms[frame_idx].start = camera_to_world_start;
		dataset.xforms[frame_idx].end = camera_to_world_end;
		dataset.metadata[frame_idx].rolling_shutter = rolling_shutter;
		dataset.update_metadata(frame_idx, frame_idx + 1);

		cam_rot_offset[frame_idx].reset_state();
		cam_pos_offset[frame_idx].reset_state();
		cam_exposure[frame_idx].reset_state();
		update_transforms(frame_idx, frame_idx + 1);
	}

	void Testbed::Nerf::Training::set_camera_extrinsics(int frame_idx, mat4x3 camera_to_world, bool convert_to_ngp) {
		set_camera_extrinsics_rolling_shutter(frame_idx, camera_to_world, camera_to_world, vec4(0.0f), convert_to_ngp);
	}

	void Testbed::Nerf::Training::reset_camera_extrinsics() {
		for (auto&& opt : cam_rot_offset) {
			opt.reset_state();
		}

		for (auto&& opt : cam_pos_offset) {
			opt.reset_state();
		}

		for (auto&& opt : cam_exposure) {
			opt.reset_state();
		}
	}

	void Testbed::Nerf::Training::export_camera_extrinsics(const fs::path& path, bool export_extrinsics_in_quat_format) {
		tlog::info() << "Saving a total of " << n_images_for_training << " poses to " << path.str();
		nlohmann::json trajectory;
		for (int i = 0; i < n_images_for_training; ++i) {
			nlohmann::json frame{ {"id", i} };

			const mat4x3 p_nerf = get_camera_extrinsics(i);
			if (export_extrinsics_in_quat_format) {
				// Assume 30 fps
				frame["time"] = i * 0.033f;
				// Convert the pose from NeRF to Quaternion format.
				const mat3 conv_coords_l{
					 0.f,   0.f,  -1.f,
					 1.f,   0.f,   0.f,
					 0.f,  -1.f,   0.f,
				};
				const mat4 conv_coords_r{
					1.f,  0.f,  0.f,  0.f,
					0.f, -1.f,  0.f,  0.f,
					0.f,  0.f, -1.f,  0.f,
					0.f,  0.f,  0.f,  1.f,
				};

				const mat4x3 p_quat = conv_coords_l * p_nerf * conv_coords_r;

				const quat rot_q = mat3(p_quat);
				frame["q"] = rot_q;
				frame["t"] = p_quat[3];
			}
			else {
				frame["transform_matrix"] = p_nerf;
			}

			trajectory.emplace_back(frame);
		}

		std::ofstream file{ native_string(path) };
		file << std::setw(2) << trajectory << std::endl;
	}

	mat4x3 Testbed::Nerf::Training::get_camera_extrinsics(int frame_idx) {
		if (frame_idx < 0 || frame_idx >= dataset.n_images) {
			return mat4x3::identity();
		}
		return dataset.ngp_matrix_to_nerf(transforms[frame_idx].start);
	}

	void Testbed::Nerf::Training::update_transforms(int first, int last) {
		if (last < 0) {
			last = dataset.n_images;
		}

		if (last > dataset.n_images) {
			last = dataset.n_images;
		}

		int n = last - first;
		if (n <= 0) {
			return;
		}

		if (transforms.size() < last) {
			transforms.resize(last);
		}

		for (uint32_t i = 0; i < n; ++i) {
			auto xform = dataset.xforms[i + first];
			float det_start = determinant(mat3(xform.start));
			float det_end = determinant(mat3(xform.end));
			if (distance(det_start, 1.0f) > 0.01f || distance(det_end, 1.0f) > 0.01f) {
				tlog::warning() << "Rotation of camera matrix in frame " << i + first << " has a scaling component (determinant!=1).";
				tlog::warning() << "Normalizing the matrix. This hints at an issue in your data generation pipeline and should be fixed.";

				xform.start[0] /= std::cbrt(det_start); xform.start[1] /= std::cbrt(det_start); xform.start[2] /= std::cbrt(det_start);
				xform.end[0] /= std::cbrt(det_end);   xform.end[1] /= std::cbrt(det_end);   xform.end[2] /= std::cbrt(det_end);
				dataset.xforms[i + first] = xform;
			}

			mat3 rot = rotmat(cam_rot_offset[i + first].variable());
			auto rot_start = rot * mat3(xform.start);
			auto rot_end = rot * mat3(xform.end);
			xform.start = mat4x3(rot_start[0], rot_start[1], rot_start[2], xform.start[3]);
			xform.end = mat4x3(rot_end[0], rot_end[1], rot_end[2], xform.end[3]);

			xform.start[3] += cam_pos_offset[i + first].variable();
			xform.end[3] += cam_pos_offset[i + first].variable();
			transforms[i + first] = xform;
		}

		transforms_gpu.enlarge(last);
		CUDA_CHECK_THROW(cudaMemcpy(transforms_gpu.data() + first, transforms.data() + first, n * sizeof(TrainingXForm), cudaMemcpyHostToDevice));
	}

	void Testbed::create_empty_nerf_dataset(size_t n_images, int aabb_scale, bool is_hdr) {
		m_data_path = {};
		set_mode(ETestbedMode::Nerf);
		m_nerf.training.dataset = ngp::create_empty_nerf_dataset(n_images, aabb_scale, is_hdr);
		load_nerf(m_data_path);
		m_nerf.training.n_images_for_training = 0;
		m_training_data_available = true;
	}

	void Testbed::load_nerf_post() { // moved the second half of load_nerf here
		m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

		m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

		m_nerf.training.dataset.update_metadata();

		m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
		m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

		m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
		m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
		m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
		m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

		m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
		m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

		m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
		m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
		m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

		m_nerf.training.cam_focal_length_gradient = vec2(0.0f);
		m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

		m_nerf.reset_extra_dims(m_rng);
		m_nerf.training.optimize_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0;

		if (m_nerf.training.dataset.has_rays) {
			m_nerf.training.near_distance = 0.0f;
		}

		// Perturbation of the training cameras -- for debugging the online extrinsics learning code
		// float perturb_amount = 0.01f;
		// if (perturb_amount > 0.f) {
		// 	for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
		// 		vec3 rot = (random_val_3d(m_rng) * 2.0f - 1.0f) * perturb_amount;
		// 		vec3 trans = (random_val_3d(m_rng) * 2.0f - 1.0f) * perturb_amount;
		// 		float angle = length(rot);
		// 		rot /= angle;

		// 		auto rot_start = rotmat(angle, rot) * mat3(m_nerf.training.dataset.xforms[i].start);
		// 		auto rot_end = rotmat(angle, rot) * mat3(m_nerf.training.dataset.xforms[i].end);
		// 		m_nerf.training.dataset.xforms[i].start = mat4x3(rot_start[0], rot_start[1], rot_start[2], m_nerf.training.dataset.xforms[i].start[3] + trans);
		// 		m_nerf.training.dataset.xforms[i].end = mat4x3(rot_end[0], rot_end[1], rot_end[2], m_nerf.training.dataset.xforms[i].end[3] + trans);
		// 	}
		// }

		m_nerf.training.update_transforms();

		if (!m_nerf.training.dataset.metadata.empty()) {
			m_nerf.render_lens = m_nerf.training.dataset.metadata[0].lens;
			m_screen_center = vec2(1.f) - m_nerf.training.dataset.metadata[0].principal_point;
		}

		if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
			throw std::runtime_error{ fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.", m_nerf.training.dataset.aabb_scale) };
		}

		int max_aabb_scale = 1 << (NERF_CASCADES() - 1);
		if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
			throw std::runtime_error{ fmt::format(
				"NeRF dataset must have `aabb_scale <= {}`, but is {}. "
				"You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
				max_aabb_scale, m_nerf.training.dataset.aabb_scale
			) };
		}

		m_aabb = BoundingBox{ vec3(0.5f), vec3(0.5f) };
		m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES() - 1), m_nerf.training.dataset.aabb_scale));
		m_raw_aabb = m_aabb;
		m_render_aabb = m_aabb;
		m_render_aabb_to_local = m_nerf.training.dataset.render_aabb_to_local;
		if (!m_nerf.training.dataset.render_aabb.is_empty()) {
			m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
		}

		m_nerf.max_cascade = 0;
		while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
			++m_nerf.max_cascade;
		}

		// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
		// stepping in larger scenes.
		m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

		m_up_dir = m_nerf.training.dataset.up;
	}

	void Testbed::load_nerf(const fs::path& data_path) {
		if (!data_path.empty()) {
			std::vector<fs::path> json_paths;
			if (data_path.is_directory()) {
				for (const auto& path : fs::directory{ data_path }) {
					if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
						json_paths.emplace_back(path);
					}
				}
			}
			else if (equals_case_insensitive(data_path.extension(), "json")) {
				json_paths.emplace_back(data_path);
			}
			else {
				throw std::runtime_error{ "NeRF data path must either be a json file or a directory containing json files." };
			}

			const auto prev_aabb_scale = m_nerf.training.dataset.aabb_scale;

			m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);

			// Check if the NeRF network has been previously configured.
			// If it has not, don't reset it.
			if (m_nerf.training.dataset.aabb_scale != prev_aabb_scale && m_nerf_network) {
				// The AABB scale affects network size indirectly. If it changed after loading,
				// we need to reset the previously configured network to keep a consistent internal state.
				reset_network();
			}
		}

		load_nerf_post();
	}

	void Testbed::update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream) {
		const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);

		m_nerf.density_grid.resize(n_elements);

		const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;
		//printf(" n_density_grid_samples : %u \n", n_density_grid_samples);

		const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

		GPUMemoryArena::Allocation alloc;
		auto scratch = allocate_workspace_and_distribute<
			NerfPosition,       // positions at which the NN will be queried for density evaluation
			uint32_t,           // indices of corresponding density grid cells
			float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
			network_precision_t // output of the MLP before being converted to densities.
		>(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

		NerfPosition* density_grid_positions = std::get<0>(scratch);
		uint32_t* density_grid_indices = std::get<1>(scratch);
		float* density_grid_tmp = std::get<2>(scratch);
		network_precision_t* mlp_out = std::get<3>(scratch);

		if (m_training_step == 0 || m_nerf.training.n_images_for_training != m_nerf.training.n_images_for_training_prev) {
			m_nerf.training.n_images_for_training_prev = m_nerf.training.n_images_for_training;
			if (m_training_step == 0) {
				m_nerf.density_grid_ema_step = 0;
			}
			// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
			if (!m_nerf.training.dataset.has_rays) {
				linear_kernel(mark_untrained_density_grid, 0, stream, n_elements, m_nerf.density_grid.data(),
					m_nerf.training.n_images_for_training,
					m_nerf.training.dataset.metadata_gpu.data(),
					m_nerf.training.transforms_gpu.data(),
					m_training_step == 0
				);
			}
			else {
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float) * n_elements, stream));
			}
		}

		uint32_t n_steps = 1;
		for (uint32_t i = 0; i < n_steps; ++i) {
			CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float) * n_elements, stream));

			linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
				n_uniform_density_grid_samples,
				m_nerf.training.density_grid_rng,
				m_nerf.density_grid_ema_step,
				m_aabb,
				m_nerf.density_grid.data(),
				density_grid_positions,
				density_grid_indices,
				m_nerf.max_cascade + 1,
				-0.01f
			);
			m_nerf.training.density_grid_rng.advance();

			linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
				n_nonuniform_density_grid_samples,
				m_nerf.training.density_grid_rng,
				m_nerf.density_grid_ema_step,
				m_aabb,
				m_nerf.density_grid.data(),
				density_grid_positions + n_uniform_density_grid_samples,
				density_grid_indices + n_uniform_density_grid_samples,
				m_nerf.max_cascade + 1,
				NERF_MIN_OPTICAL_THICKNESS()
			);
			m_nerf.training.density_grid_rng.advance();

			// Evaluate density at the spawned locations in batches.
			// Otherwise, we can exhaust the maximum index range of cutlass
			size_t batch_size = NERF_GRID_N_CELLS() * 2;

			for (size_t i = 0; i < n_density_grid_samples; i += batch_size) {
				batch_size = std::min(batch_size, n_density_grid_samples - i);

				GPUMatrix<network_precision_t, RM> density_matrix(mlp_out + i, padded_output_width, batch_size);
				GPUMatrix<float> density_grid_position_matrix((float*)(density_grid_positions + i), sizeof(NerfPosition) / sizeof(float), batch_size);
				m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);
			}

			linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp, m_nerf.rgb_activation, m_nerf.density_activation);
			linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

			++m_nerf.density_grid_ema_step;
		}

		update_density_grid_mean_and_bitfield(stream);
	}

	void Testbed::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
		const uint32_t n_elements = NERF_GRID_N_CELLS();

		size_t size_including_mips = grid_mip_offset(NERF_CASCADES()) / 8;
		m_nerf.density_grid_bitfield.enlarge(size_including_mips);
		m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
		reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__(float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

		linear_kernel(grid_to_bitfield, 0, stream, n_elements / 8 * NERF_CASCADES(), n_elements / 8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

		for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
			linear_kernel(bitfield_max_pool, 0, stream, n_elements / 64, m_nerf.get_density_grid_bitfield_mip(level - 1), m_nerf.get_density_grid_bitfield_mip(level));
		}

		set_all_devices_dirty();
	}

	__global__ void mark_density_grid_in_sphere_empty_kernel(const uint32_t n_elements, float* density_grid, vec3 pos, float radius) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		// Random position within that cellq
		uint32_t level = i / NERF_GRID_N_CELLS();
		uint32_t pos_idx = i % NERF_GRID_N_CELLS();

		uint32_t x = morton3D_invert(pos_idx >> 0);
		uint32_t y = morton3D_invert(pos_idx >> 1);
		uint32_t z = morton3D_invert(pos_idx >> 2);

		float cell_radius = scalbnf(SQRT3(), level) / NERF_GRIDSIZE();
		vec3 cell_pos = ((vec3{ (float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f }) / (float)NERF_GRIDSIZE() - 0.5f) * scalbnf(1.0f, level) + 0.5f;

		// Disable if the cell touches the sphere (conservatively, by bounding the cell with a sphere)
		if (distance(pos, cell_pos) < radius + cell_radius) {
			density_grid[i] = -1.0f;
		}
	}

	void Testbed::mark_density_grid_in_sphere_empty(const vec3& pos, float radius, cudaStream_t stream) {
		const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);
		if (m_nerf.density_grid.size() != n_elements) {
			return;
		}

		linear_kernel(mark_density_grid_in_sphere_empty_kernel, 0, stream, n_elements, m_nerf.density_grid.data(), pos, radius);

		update_density_grid_mean_and_bitfield(stream);
	}

	void Testbed::NerfCounters::prepare_for_training_steps(cudaStream_t stream) {
		numsteps_counter.enlarge(1);
		numsteps_counter_compacted.enlarge(1);
		loss.enlarge(rays_per_batch);
		CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
		CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter_compacted.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
		CUDA_CHECK_THROW(cudaMemsetAsync(loss.data(), 0, sizeof(float) * rays_per_batch, stream));
	}

	float Testbed::NerfCounters::update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
		std::vector<uint32_t> counter_cpu(1);
		std::vector<uint32_t> compacted_counter_cpu(1);
		numsteps_counter.copy_to_host(counter_cpu);
		numsteps_counter_compacted.copy_to_host(compacted_counter_cpu);
		measured_batch_size = 0;
		measured_batch_size_before_compaction = 0;

		if (counter_cpu[0] == 0 || compacted_counter_cpu[0] == 0) {
			return 0.f;
		}

		measured_batch_size_before_compaction = counter_cpu[0];
		measured_batch_size = compacted_counter_cpu[0];

		float loss_scalar = 0.0;
		if (get_loss_scalar) {
			loss_scalar = reduce_sum(loss.data(), rays_per_batch, stream) * (float)measured_batch_size / (float)target_batch_size;
		}

		rays_per_batch = (uint32_t)((float)rays_per_batch * (float)target_batch_size / (float)measured_batch_size);
		rays_per_batch = std::min(next_multiple(rays_per_batch, BATCH_SIZE_GRANULARITY), 1u << 18);

		return loss_scalar;
	}

	void Testbed::train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
		if (m_nerf.training.n_images_for_training == 0) {
			return;
		}

		if (m_nerf.training.include_sharpness_in_error) {
			size_t n_cells = NERF_GRID_N_CELLS() * NERF_CASCADES();
			if (m_nerf.training.sharpness_grid.size() < n_cells) {
				m_nerf.training.sharpness_grid.enlarge(NERF_GRID_N_CELLS() * NERF_CASCADES());
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
			}

			if (m_training_step == 0) {
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
			}
			else {
				linear_kernel(decay_sharpness_grid_nerf, 0, stream, m_nerf.training.sharpness_grid.size(), 0.95f, m_nerf.training.sharpness_grid.data());
			}
		}
		m_nerf.training.counters_rgb.prepare_for_training_steps(stream);

		if (m_nerf.training.n_steps_since_cam_update == 0) {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float) * m_distortion.map->n_params(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float) * m_distortion.map->n_params(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));
		}

		bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
		uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
		if (train_extra_dims) {
			uint32_t n = n_extra_dims * m_nerf.training.n_images_for_training;
			m_nerf.training.extra_dims_gradient_gpu.enlarge(n);
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.extra_dims_gradient_gpu.data(), 0, m_nerf.training.extra_dims_gradient_gpu.get_bytes(), stream));
		}

		if (m_nerf.training.n_steps_since_error_map_update == 0 && !m_nerf.training.dataset.metadata.empty()) {
			uint32_t n_samples_per_image = (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.counters_rgb.rays_per_batch) / m_nerf.training.dataset.n_images;
			ivec2 res = m_nerf.training.dataset.metadata[0].resolution;
			m_nerf.training.error_map.resolution = min(ivec2((int)(std::sqrt(std::sqrt((float)n_samples_per_image)) * 3.5f)), res);
			m_nerf.training.error_map.data.resize(product(m_nerf.training.error_map.resolution) * m_nerf.training.dataset.n_images);
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
		}

		float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
		if (envmap_gradient) {
			CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float) * m_envmap.envmap->n_params(), stream));
		}

		train_nerf_step(target_batch_size, m_nerf.training.counters_rgb, stream);


		m_trainer->optimizer_step(stream, LOSS_SCALE());

		++m_training_step;

		if (envmap_gradient) {
			m_envmap.trainer->optimizer_step(stream, LOSS_SCALE());
		}

		float loss_scalar = m_nerf.training.counters_rgb.update_after_training(target_batch_size, get_loss_scalar, stream);
		bool zero_records = m_nerf.training.counters_rgb.measured_batch_size == 0;
		if (get_loss_scalar) {
			m_loss_scalar.update(loss_scalar);
		}

		if (zero_records) {
			m_loss_scalar.set(0.f);
			tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
			m_train = false;
		}

		// Compute CDFs from the error map
		m_nerf.training.n_steps_since_error_map_update += 1;
		// This is low-overhead enough to warrant always being on.
		// It makes for useful visualizations of the training error.
		bool accumulate_error = true;
		if (accumulate_error && m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates) {
			m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
			m_nerf.training.error_map.cdf_x_cond_y.resize(product(m_nerf.training.error_map.cdf_resolution) * m_nerf.training.dataset.n_images);
			m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y * m_nerf.training.dataset.n_images);
			m_nerf.training.error_map.cdf_img.resize(m_nerf.training.dataset.n_images);

			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(), 0, m_nerf.training.error_map.cdf_x_cond_y.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

			const dim3 threads = { 16, 8, 1 };
			const dim3 blocks = { div_round_up((uint32_t)m_nerf.training.error_map.cdf_resolution.y, threads.x), div_round_up((uint32_t)m_nerf.training.dataset.n_images, threads.y), 1 };
			construct_cdf_2d << <blocks, threads, 0, stream >> > (
				m_nerf.training.dataset.n_images, m_nerf.training.error_map.cdf_resolution.y, m_nerf.training.error_map.cdf_resolution.x,
				m_nerf.training.error_map.data.data(),
				m_nerf.training.error_map.cdf_x_cond_y.data(),
				m_nerf.training.error_map.cdf_y.data()
				);
			linear_kernel(construct_cdf_1d, 0, stream,
				m_nerf.training.dataset.n_images,
				m_nerf.training.error_map.cdf_resolution.y,
				m_nerf.training.error_map.cdf_y.data(),
				m_nerf.training.error_map.cdf_img.data()
			);

			// Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
			m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
			m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
			std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
			float cum = 0;
			for (float& f : cdf_img_cpu) {
				cum += f;
				f = cum;
			}
			float norm = 1.0f / cum;
			for (size_t i = 0; i < cdf_img_cpu.size(); ++i) {
				constexpr float MIN_PMF = 0.1f;
				m_nerf.training.error_map.pmf_img_cpu[i] = (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm + MIN_PMF / (float)m_nerf.training.dataset.n_images;
				cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm + MIN_PMF * (float)(i + 1) / (float)m_nerf.training.dataset.n_images;
			}
			m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

			// Reset counters and decrease update rate.
			m_nerf.training.n_steps_since_error_map_update = 0;
			m_nerf.training.n_rays_since_error_map_update = 0;
			m_nerf.training.error_map.is_cdf_valid = true;

			m_nerf.training.n_steps_between_error_map_updates = (uint32_t)(m_nerf.training.n_steps_between_error_map_updates * 1.5f);
		}

		// Get extrinsics gradients
		m_nerf.training.n_steps_since_cam_update += 1;

		if (train_extra_dims) {
			std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
			m_nerf.training.extra_dims_gradient_gpu.copy_to_host(extra_dims_gradient);

			// Optimization step
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				std::vector<float> gradient(n_extra_dims);
				for (uint32_t j = 0; j < n_extra_dims; ++j) {
					gradient[j] = extra_dims_gradient[i * n_extra_dims + j] / LOSS_SCALE();
				}

				//float l2_reg = 1e-4f;
				//gradient += m_nerf.training.extra_dims_opt[i].variable() * l2_reg;

				m_nerf.training.extra_dims_opt[i].set_learning_rate(m_optimizer->learning_rate());
				m_nerf.training.extra_dims_opt[i].step(gradient);
			}

			m_nerf.training.update_extra_dims();
		}

		bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
		if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
			float per_camera_loss_scale = (float)m_nerf.training.n_images_for_training / LOSS_SCALE() / (float)m_nerf.training.n_steps_between_cam_updates;

			if (m_nerf.training.optimize_extrinsics) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

				// Optimization step
				for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
					vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
					vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

					float l2_reg = m_nerf.training.extrinsic_l2_reg;
					pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
					rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

					m_nerf.training.cam_pos_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_pos_offset[i].step() / 128)), m_optimizer->learning_rate() / 1000.0f));
					m_nerf.training.cam_rot_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_rot_offset[i].step() / 128)), m_optimizer->learning_rate() / 1000.0f));

					m_nerf.training.cam_pos_offset[i].step(pos_gradient);
					m_nerf.training.cam_rot_offset[i].step(rot_gradient);
				}

				m_nerf.training.update_transforms();
			}

			if (m_nerf.training.optimize_distortion) {
				linear_kernel(safe_divide, 0, stream,
					m_distortion.map->n_params(),
					m_distortion.map->gradients(),
					m_distortion.map->gradient_weights()
				);
				m_distortion.trainer->optimizer_step(stream, LOSS_SCALE() * (float)m_nerf.training.n_steps_between_cam_updates);
			}

			if (m_nerf.training.optimize_focal_length) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(&m_nerf.training.cam_focal_length_gradient, m_nerf.training.cam_focal_length_gradient_gpu.data(), m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
				vec2 focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
				float l2_reg = m_nerf.training.intrinsic_l2_reg;
				focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
				m_nerf.training.cam_focal_length_offset.set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_focal_length_offset.step() / 128)), m_optimizer->learning_rate() / 1000.0f));
				m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
				m_nerf.training.dataset.update_metadata();
			}

			if (m_nerf.training.optimize_exposure) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(), m_nerf.training.cam_exposure_gradient_gpu.data(), m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

				vec3 mean_exposure = vec3(0.0f);

				// Optimization step
				for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
					vec3 gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

					float l2_reg = m_nerf.training.exposure_l2_reg;
					gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

					m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
					m_nerf.training.cam_exposure[i].step(gradient);

					mean_exposure += m_nerf.training.cam_exposure[i].variable();
				}

				mean_exposure /= (float)m_nerf.training.n_images_for_training;

				// Renormalize
				std::vector<vec3> cam_exposures(m_nerf.training.n_images_for_training);
				for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
					cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
				}

				CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(), cam_exposures.data(), m_nerf.training.n_images_for_training * sizeof(vec3), cudaMemcpyHostToDevice, stream));
			}

			m_nerf.training.n_steps_since_cam_update = 0;
		}
	}

	void Testbed::train_nerf_step(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
		const uint32_t padded_output_width = m_network->padded_output_width();
		const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
		const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

		GPUMemoryArena::Allocation alloc;
		auto scratch = allocate_workspace_and_distribute<
			uint32_t, // ray_indices
			Ray, // rays
			uint32_t, // numsteps
			float, // coords
			float, // max_level
			network_precision_t, // mlp_out
			network_precision_t, // dloss_dmlp_out
			float, // coords_compacted
			float, // coords_gradient
			float, // max_level_compacted
			uint32_t // ray_counter
		>(
			stream, &alloc,
			counters.rays_per_batch,
			counters.rays_per_batch,
			counters.rays_per_batch * 2,
			max_samples * floats_per_coord,
			max_samples,
			std::max(target_batch_size, max_samples) * padded_output_width,
			target_batch_size * padded_output_width,
			target_batch_size * floats_per_coord,
			target_batch_size * floats_per_coord,
			target_batch_size,
			1
			);

		// TODO: C++17 structured binding
		uint32_t* ray_indices = std::get<0>(scratch);
		Ray* rays_unnormalized = std::get<1>(scratch);
		uint32_t* numsteps = std::get<2>(scratch);
		float* coords = std::get<3>(scratch);
		float* max_level = std::get<4>(scratch);
		network_precision_t* mlp_out = std::get<5>(scratch);
		network_precision_t* dloss_dmlp_out = std::get<6>(scratch);
		float* coords_compacted = std::get<7>(scratch);
		float* coords_gradient = std::get<8>(scratch);
		float* max_level_compacted = std::get<9>(scratch);
		uint32_t* ray_counter = std::get<10>(scratch);

		uint32_t max_inference;
		if (counters.measured_batch_size_before_compaction == 0) {
			counters.measured_batch_size_before_compaction = max_inference = max_samples;
		}
		else {
			max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), BATCH_SIZE_GRANULARITY);
		}

		GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
		GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

		GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

		if (m_training_step == 0) {
			counters.n_rays_total = 0;
		}

		uint32_t n_rays_total = counters.n_rays_total;
		counters.n_rays_total += counters.rays_per_batch;
		m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

		// If we have an envmap, prepare its gradient buffer
		float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;

		bool sample_focal_plane_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
		bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
		bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
		// This is low-overhead enough to warrant always being on.
		// It makes for useful visualizations of the training error.
		bool accumulate_error = true;

		CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));

		auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

		{
			linear_kernel(generate_training_samples_nerf, 0, stream,
				counters.rays_per_batch,
				m_aabb,
				max_inference,
				n_rays_total,
				m_rng,
				ray_counter,
				counters.numsteps_counter.data(),
				ray_indices,
				rays_unnormalized,
				numsteps,
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
				m_nerf.training.n_images_for_training,
				m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				m_nerf.density_grid_bitfield.data(),
				m_nerf.max_cascade,
				m_max_level_rand_training,
				max_level,
				m_nerf.training.snap_to_pixel_centers,
				m_nerf.training.train_envmap,
				m_nerf.cone_angle_constant,
				m_distortion.view(),
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
				sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
				m_nerf.training.error_map.cdf_resolution,
				m_nerf.training.extra_dims_gpu.data(),
				m_nerf_network->n_extra_dims()
			);

			if (hg_enc) {
				hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
			}

			GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
			GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
			m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

			if (hg_enc) {
				hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
			}

			linear_kernel(compute_loss_kernel_train_nerf, 0, stream,
				counters.rays_per_batch,
				m_aabb,
				n_rays_total,
				m_rng,
				target_batch_size,
				ray_counter,
				LOSS_SCALE(),
				padded_output_width,
				m_envmap.view(),
				envmap_gradient,
				m_envmap.resolution,
				m_envmap.loss_type,
				m_background_color.rgb(),
				m_color_space,
				m_nerf.training.random_bg_color,
				m_nerf.training.linear_colors,
				m_nerf.training.n_images_for_training,
				m_nerf.training.dataset.metadata_gpu.data(),
				mlp_out,
				counters.numsteps_counter_compacted.data(),
				ray_indices,
				rays_unnormalized,
				numsteps,
				PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
				dloss_dmlp_out,
				m_nerf.training.loss_type,
				m_nerf.training.depth_loss_type,
				counters.loss.data(),
				m_max_level_rand_training,
				max_level_compacted,
				m_nerf.rgb_activation,
				m_nerf.density_activation,
				m_nerf.training.snap_to_pixel_centers,
				accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
				sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
				m_nerf.training.error_map.resolution,
				m_nerf.training.error_map.cdf_resolution,
				include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
				m_nerf.training.dataset.sharpness_resolution,
				m_nerf.training.sharpness_grid.data(),
				m_nerf.density_grid.data(),
				m_nerf.density_grid_mean.data(),
				m_nerf.max_cascade,
				m_nerf.training.cam_exposure_gpu.data(),
				m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
				m_nerf.training.depth_supervision_lambda,
				m_nerf.training.near_distance
			);
		}

		fill_rollover_and_rescale<network_precision_t> << <n_blocks_linear(target_batch_size * padded_output_width), N_THREADS_LINEAR, 0, stream >> > (
			target_batch_size, padded_output_width, counters.numsteps_counter_compacted.data(), dloss_dmlp_out
			);
		fill_rollover<float> << <n_blocks_linear(target_batch_size * floats_per_coord), N_THREADS_LINEAR, 0, stream >> > (
			target_batch_size, floats_per_coord, counters.numsteps_counter_compacted.data(), (float*)coords_compacted
			);
		fill_rollover<float> << <n_blocks_linear(target_batch_size), N_THREADS_LINEAR, 0, stream >> > (
			target_batch_size, 1, counters.numsteps_counter_compacted.data(), max_level_compacted
			);

		bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length;
		bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
		bool prepare_input_gradients = train_camera || train_extra_dims;
		GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

		m_trainer->training_step(stream, compacted_coords_matrix, {}, nullptr, false, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, GradientMode::Overwrite, &gradient_matrix);

		if (train_extra_dims) {
			// Compute extra-dim gradients
			linear_kernel(compute_extra_dims_gradient_train_nerf, 0, stream,
				counters.rays_per_batch,
				n_rays_total,
				ray_counter,
				m_nerf.training.extra_dims_gradient_gpu.data(),
				m_nerf.training.dataset.n_extra_dims(),
				m_nerf.training.n_images_for_training,
				ray_indices,
				numsteps,
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
				sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr
			);
		}

		if (train_camera) {
			// Compute camera gradients
			linear_kernel(compute_cam_gradient_train_nerf, 0, stream,
				counters.rays_per_batch,
				n_rays_total,
				m_rng,
				m_aabb,
				ray_counter,
				m_nerf.training.transforms_gpu.data(),
				m_nerf.training.snap_to_pixel_centers,
				m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_pos_gradient_gpu.data() : nullptr,
				m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_rot_gradient_gpu.data() : nullptr,
				m_nerf.training.n_images_for_training,
				m_nerf.training.dataset.metadata_gpu.data(),
				ray_indices,
				rays_unnormalized,
				numsteps,
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
				m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
				m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
				m_distortion.resolution,
				m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
				sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
				m_nerf.training.error_map.cdf_resolution
			);
		}

		m_rng.advance();

		if (hg_enc) {
			hg_enc->set_max_level_gpu(nullptr);
		}
	}


	void Testbed::training_prep_nerf(uint32_t batch_size, cudaStream_t stream) {
		if (m_nerf.training.n_images_for_training == 0) {
			return;
		}


		float alpha = m_nerf.training.density_grid_decay;//
		uint32_t n_cascades = m_nerf.max_cascade + 1;

		if (m_training_step < 256) {
			update_density_grid_nerf(alpha, NERF_GRID_N_CELLS() * n_cascades, 0, stream);//128*128*128
		}
		else {
			update_density_grid_nerf(alpha, NERF_GRID_N_CELLS() / 4 * n_cascades, NERF_GRID_N_CELLS() / 4 * n_cascades, stream);
		}
	}

	void Testbed::optimise_mesh_step(uint32_t n_steps) {
		uint32_t n_verts = (uint32_t)m_mesh.verts.size();
		if (!n_verts) {
			return;
		}

		const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();
		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
		const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);
		GPUMemory<float> coords(n_verts * floats_per_coord);
		GPUMemory<network_precision_t> mlp_out(n_verts * padded_output_width);

		GPUMatrix<float> positions_matrix((float*)coords.data(), floats_per_coord, n_verts);
		GPUMatrix<network_precision_t, RM> density_matrix(mlp_out.data(), padded_output_width, n_verts);

		const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(m_stream.get());

		for (uint32_t i = 0; i < n_steps; ++i) {
			linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_stream.get(),
				n_verts,
				m_aabb,
				m_mesh.verts.data(),
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords.data(), 1, 0, extra_stride),
				extra_dims_gpu
			);

			// For each optimizer step, we need the density at the given pos...
			m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
			// ...as well as the input gradient w.r.t. density, which we will store in the nerf coords.
			m_nerf_network->input_gradient(m_stream.get(), 3, positions_matrix, positions_matrix);
			// and the 1ring centroid for laplacian smoothing
			compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);

			// With these, we can compute a gradient that points towards the threshold-crossing of density...
			compute_mesh_opt_gradients(
				m_mesh.thresh,
				m_mesh.verts,
				m_mesh.vert_normals,
				m_mesh.verts_smoothed,
				mlp_out.data(),
				floats_per_coord,
				(const float*)coords.data(),
				m_mesh.verts_gradient,
				m_mesh.smooth_amount,
				m_mesh.density_amount,
				m_mesh.inflate_amount
			);

			// ...that we can pass to the optimizer.
			m_mesh.verts_optimizer->step(m_stream.get(), 1.0f, (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
		}
	}

	void Testbed::compute_mesh_vertex_colors() {
		uint32_t n_verts = (uint32_t)m_mesh.verts.size();
		if (!n_verts) {
			return;
		}

		m_mesh.vert_colors.resize(n_verts);
		m_mesh.vert_colors.memset(0);

		if (m_testbed_mode == ETestbedMode::Nerf) {
			const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(m_stream.get());

			const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
			const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);
			GPUMemory<float> coords(n_verts * floats_per_coord);
			GPUMemory<float> mlp_out(n_verts * 4);

			GPUMatrix<float> positions_matrix((float*)coords.data(), floats_per_coord, n_verts);
			GPUMatrix<float> color_matrix(mlp_out.data(), 4, n_verts);
			linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_stream.get(), n_verts, m_aabb, m_mesh.verts.data(), PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords.data(), 1, 0, extra_stride), extra_dims_gpu);
			m_network->inference(m_stream.get(), positions_matrix, color_matrix);
			linear_kernel(extract_srgb_with_activation, 0, m_stream.get(), n_verts * 3, 3, mlp_out.data(), (float*)m_mesh.vert_colors.data(), m_nerf.rgb_activation, m_nerf.training.linear_colors);
		}
	}

	GPUMemory<float> Testbed::get_density_on_grid(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local) {
		const uint32_t n_elements = (res3d.x * res3d.y * res3d.z);
		GPUMemory<float> density(n_elements);

		const uint32_t batch_size = std::min(n_elements, 1u << 20);
		bool nerf_mode = m_testbed_mode == ETestbedMode::Nerf;

		const uint32_t padded_output_width = nerf_mode ? m_nerf_network->padded_density_output_width() : m_network->padded_output_width();

		GPUMemoryArena::Allocation alloc;
		auto scratch = allocate_workspace_and_distribute<
			NerfPosition,
			network_precision_t
		>(m_stream.get(), &alloc, n_elements, batch_size * padded_output_width);

		NerfPosition* positions = std::get<0>(scratch);
		network_precision_t* mlp_out = std::get<1>(scratch);

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res3d.x, threads.x), div_round_up((uint32_t)res3d.y, threads.y), div_round_up((uint32_t)res3d.z, threads.z) };

		BoundingBox unit_cube = BoundingBox{ vec3(0.0f), vec3(1.0f) };
		generate_grid_samples_nerf_uniform << <blocks, threads, 0, m_stream.get() >> > (res3d, m_nerf.density_grid_ema_step, aabb, render_aabb_to_local, nerf_mode ? m_aabb : unit_cube, positions);

		// Only process 1m elements at a time
		for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
			uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

			GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, local_batch_size);

			GPUMatrix<float> positions_matrix((float*)(positions + offset), sizeof(NerfPosition) / sizeof(float), local_batch_size);
			if (nerf_mode) {
				m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
			}
			else {
				m_network->inference_mixed_precision(m_stream.get(), positions_matrix, density_matrix);
			}
			linear_kernel(grid_samples_half_to_float, 0, m_stream.get(),
				local_batch_size,
				m_aabb,
				density.data() + offset, //+ axis_step * n_elements,
				mlp_out,
				m_nerf.density_activation,
				positions + offset,
				nerf_mode ? m_nerf.density_grid.data() : nullptr,
				m_nerf.max_cascade
			);


		}
		//std::vector<float> dencity_cpu;
		//dencity_cpu.resize(density.size());
		//density.copy_to_host(dencity_cpu);//cpu 메모리에 값 옮김
		//float max = 0;
		//for (size_t i = 0; i < dencity_cpu.size(); ++i) {
		//	max = __max(max, dencity_cpu[i]);
		//	//printf("from gpu density[%zu]: %f\n", i, dencity_cpu[i]);
		//}
		//printf("density grid max density : %f  !!\n", max);
		//printf(" density_cpu size : %d\n ", dencity_cpu.size());


		//float* densityData = dencity_cpu.data();
		//const int dataSize = 256 * 256 * 256;

		//// Open a binary file for writing
		//std::ofstream outFile("density_data.den", std::ios::binary);

		//// Check if the file is opened successfully
		//if (!outFile.is_open()) {
		//	std::cerr << "Failed to open the file for writing." << std::endl;
		//	return 1;
		//}

		//// Write the data to the file
		//outFile.write(reinterpret_cast<char*>(densityData), dataSize * sizeof(float));

		//// Close the file
		//outFile.close();

		return density;
	}

	__global__ void filter_with_occupancy(const uint32_t n_elements, float* pos, const uint32_t floats_per_coord, const uint8_t* density_grid_bitfield, float* rgbsigma) {
		const uint32_t point_id = threadIdx.x + blockIdx.x * blockDim.x;
		if (point_id >= n_elements) return;
		const vec3 pos_vec{ pos[point_id * floats_per_coord], pos[point_id * floats_per_coord + 1], pos[point_id * floats_per_coord + 2] };
		const uint32_t mip = mip_from_pos(pos_vec);
		if (!density_grid_occupied_at(pos_vec, density_grid_bitfield, mip)) {
#pragma unroll
			for (int i = 0; i < 4; i++) { // sigma=0 would be enough, but all to 0 compress better
				rgbsigma[point_id * 4 + i] = 0.f;
			}
		}
	}



	void post_process_volume(vec4* volume_data, uint32_t volume_size) {
		const float density_threshold = 0.5f; // 임의의 값으로 설정
		printf(" local batch size : %d \n", volume_size);
		for (uint32_t i = 0; i < volume_size; ++i) {
			// 볼륨 데이터의 밀도가 임계값 미만인 경우 투명하게 처리 (불필요한 장애물 제거)
			if (volume_data[i].w < density_threshold) {
				//volume_data[i].w = 0.0f; // 투명 처리 (또는 다른 처리 방법을 선택)
			}
		}
	}

	unsigned char MyTexture[512][512][3];
	float AlphaTable[256]; // Alpha 블랜딩을 위한 AlphaTable

	__constant__ float mytime[1];

	struct voxel1 {
		float r, g, b;
		float a;
	}; // new struct voxel = r,g,b,a(density)의 값을 가짐
	const int g_TFa = 0; // min alpha
	const int g_TFb = 255; // max alpha
	__device__ float4* dev_input;
	__device__ unsigned char* dev_output;
	//float4 volume[256][256][256];

	cudaTextureObject_t voltexObj;
	cudaTextureObject_t voltexOutObj;
	cudaArray* pDevArr = 0;
	cudaArray* pDevOutArr = 0;
	GLuint texID; // 텍스처 식별자를 저장할 변수

	// CUDA device variable to store the mode
	__device__ int device_mode;

	//PBD method
	__device__ __host__ class Vertex {
	public:
		float dx, dy, dz; // 정점 좌표의 이동량. 단위 시간에 따라 움직인 총량
		float px, py, pz; // 좌표
		float vx, vy, vz; // 속도
		float ax, ay, az; // 가속도
		float d = 0; //density
		float m = 1;
		float im = 1. / m; //질량mass, invMass
	};


	__device__ __host__ struct Edge {
		int m_vert[2];// 정점 배열에 들어있는 번호
		float st = 1; //stiffness
		float rl = 0; //restLength
		//float rl; //restLength
	};


	__device__ __host__ struct Tet {
		int m_vert[4];
		int m_edges[6]; //std:array<unsigned int, 6> g_edges;
		int m_faces[4]; //std::array<unsigned int, 4> m_faces;
		float st = 1.0;
		float rv; //restVolume;
		float ra; //restAngle
	};

	__device__ __host__ struct SkinSurface {
		int m_edges[3]; //따로.. 안필요하지 않나..?
		int m_vert[3];// 정점 배열에 들어있는 번호
		float st = 1; //stiffness
		float rsa = 0; //restSkinArea
	};

	class Constraint {
	public:
		std::vector<unsigned int> m_bodies;// 디버깅을 해보니 distance Constraint 의 경우 edge의 정점, volume constraint 에선 tet의 정점이 저장되어있다.

		Constraint() {
			// 절대 호출되면 안됨
			printf("Constaint 호출 에러.\n");
		}

		Constraint(const unsigned int numberOfBodies) {
			//m_bodies.resize(numberOfBodies);
		}

		unsigned int numberOfBodies() const { return static_cast<unsigned int>(m_bodies.size()); }

	};
	class DistanceConstraint : public Constraint
	{
	public:
		//static int TYPE_ID;
		float m_restLength;
		float m_stiffness;
		DistanceConstraint() :Constraint(2) {}
		DistanceConstraint(int i1, int i2, float st, float rl) :Constraint(2) {
			m_bodies.push_back(i1);
			m_bodies.push_back(i2);
			m_restLength = rl;
			m_stiffness = st;
		}

		//   DistanceConstraint() : Constraint(2) {}
		//   virtual int& getTypeId() const { return TYPE_ID; }
		//
		//   virtual bool initConstraint(SimulationModel& model, const unsigned int particle1, const unsigned int particle2, const Real stiffness);
		//   virtual bool solvePositionConstraint(SimulationModel& model, const unsigned int iter);
	};
	class VolumeConstraint : public Constraint
	{
	public:
		//static int TYPE_ID;
		float m_stiffness;
		float m_restVolume;
		VolumeConstraint() : Constraint(4) {}
		VolumeConstraint(int i1, int i2, int i3, int i4, float st, float rv) : Constraint(4) {
			m_bodies.push_back(i1);
			m_bodies.push_back(i2);
			m_bodies.push_back(i3);
			m_bodies.push_back(i4);

			m_stiffness = st;
			m_restVolume = rv;
		}


	};
	class DihedralConstraint : public Constraint
	{
	public:
		//static int TYPE_ID;
		float m_restAngle;
		float m_stiffness;

		DihedralConstraint() : Constraint(4) {}
		DihedralConstraint(int i1, int i2, int i3, int i4, float st, float ra) : Constraint(4) {
			m_bodies.push_back(i1);
			m_bodies.push_back(i2);
			m_bodies.push_back(i3);
			m_bodies.push_back(i4);

			m_stiffness = st;
			m_restAngle = ra;
		}
	};
	//thth
	class SkinTenstionConstraint : public Constraint {
	public:
		float m_restSkinArea; //안정한 길이
		float m_stiffness;
		SkinTenstionConstraint() :Constraint(3) {} // 거리제약
		SkinTenstionConstraint(int i1, int i2, int i3, float st, float rsa) :Constraint(3) {
			m_bodies.push_back(i1); //skinSurface 정점..?
			m_bodies.push_back(i2);
			m_bodies.push_back(i3);
			m_restSkinArea = rsa; //@ 무슨 길이..??
			m_stiffness = st;
		}

	};
	//class Vertex {
	//public:
	//	float dx, dy, dz; // 정점 좌표의 이동량. 단위 시간에 따라 움직인 총량
	//	float px, py, pz; // 좌표
	//	float vx, vy, vz; // 속도
	//	float ax, ay, az; // 가속도
	//
	//	float d = 0; //density
	//	float m = 1;
	//	float im = 1. / m; //질량mass, invMass
	//};
	//struct Edge {
	//	int m_vert[2];// 정점 배열에 들어있는 번호
	//	float st = 1; //stiffness
	//	float rl = 0; //restLength
	//				  //float rl; //restLength
	//};
	struct Face
	{
		// edge indices
		int m_edges[3];
		// tet indices
		int m_tets[2]; //std::array<unsigned int, 2> m_tets;
	};
	//struct Tet
	//{
	//	int m_vert[4];
	//	int m_edges[6]; //std::array<unsigned int, 6> g_edges;
	//	int m_faces[4]; //std::array<unsigned int, 4> m_faces;
	//	float st = 0.1;
	//	float rv; //restVolume
	//	float ra; //res
	//};
	////thth
	//struct SkinSurface
	//{
	//	int m_edges[3]; //따로.. 안필요하지 않나..?
	//	int m_vert[3];// 정점 배열에 들어있는 번호
	//	float st = 1; //stiffness
	//	float rsa = 0; //restSkinArea
	//
	//};

	typedef std::vector <unsigned int> ConstraintGroup;
	typedef std::vector <ConstraintGroup> ConstraintGroupVector;
	ConstraintGroupVector g_prallelizableEdgeGroups;
	ConstraintGroupVector g_prallelizableTetGroups;
	//thth
	ConstraintGroupVector g_prallelizableSkinGroups;
	typedef std::vector<Constraint> ConstraintVector;
	ConstraintVector distance_constraints;
	ConstraintVector volume_constraints;
	ConstraintVector dihedral_constraints;
	//thth
	ConstraintVector skin_constraints;
	bool m_groupsInitialized;
	Vertex vertices[PBD_X * PBD_Y * PBD_Z];
	std::vector<unsigned int> indices;


	typedef std::vector<unsigned int> Tets;
	typedef std::vector<unsigned int> Faces;
	typedef std::vector<Tet> TetData;
	typedef std::vector<Face> FaceData;
	typedef std::vector<Edge> Edges;
	//thth
	typedef std::vector<SkinSurface> SkinSurfaces;
	typedef std::vector<std::vector<unsigned int>> VerticesEdges;
	typedef std::vector<std::vector<unsigned int>> VerticesFaces;
	typedef std::vector<std::vector<unsigned int>> VerticesTets;
	VerticesEdges g_verticesEdges;

	Tets g_tetIndices;
	Faces g_faceIndices;
	Edges g_edges;
	//thth
	SkinSurfaces g_skins;

	Edge* edges; //for GPU
	//thth
	SkinSurface* skins; //for GPU
	Tet* tets; //for GPU
	TetData g_tets;

	//cudaPBD.cu 변수
	Vertex* d_vertices;
	Edge* d_edges;
	Tet* d_tets;
	SkinSurface* d_skins;

	const float dt = 0.01;
	__device__ const float eps = static_cast<float>(1e-6);

	//@소진
	float Testbed::TriangleArea3D(float p1[3], float p2[3], float p3[3]) {
		float a[3] = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
		float b[3] = { p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] };

		float crossProduct[3] = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };

		float area = 0.5 * sqrt(crossProduct[0] * crossProduct[0] + crossProduct[1] * crossProduct[1] + crossProduct[2] * crossProduct[2]);

		return area;
	}

	void Testbed::InitSkinSurface(int v1, int v2, int v3) {
		float p1[3] = { vertices[v1].px, vertices[v1].py, vertices[v1].pz };
		float p2[3] = { vertices[v2].px, vertices[v2].py, vertices[v2].pz };
		float p3[3] = { vertices[v3].px, vertices[v3].py, vertices[v3].pz };
		//지금은 사용하지 않는 부분
		float rsa = TriangleArea3D(p1, p2, p3);
		//printf("InitSkineSurface");
		SkinSurface skinSurface;
		skinSurface.m_vert[0] = v1;
		skinSurface.m_vert[1] = v2;
		skinSurface.m_vert[2] = v3;
		skinSurface.rsa = rsa * 0.5; // Adjust as needed
		//printf("%f \n", rsa);
		skinSurface.st = (vertices[v1].d + vertices[v2].d + vertices[v3].d) / 3 / 500.; // Adjust as needed

		g_skins.push_back(skinSurface);
	}
	//@소진
	void Testbed::DetectAndInitSkinSurface() {
		//float minSkinDensity = 575; //이정도 값은 되어야 배꼽형태가 명확
		//float maxSkinDensity = 1853; //이 값 부터 왼쪽 윗면 스킨 전부 검출
		//float maxSkinDensity = 1030; //근데 과연 윗면이 다 나오는게 피부인가..?
		int airVertexCount;
		int airVertexIndices[4];
		int skinVertices[3]; // 나머지 3개의 피부로 판정된 정점들을 저장할 배열
		int skinVertexCount = 0; // 피부로 gr판정된 정점의 개수
		int n = 0;
		//MakeAlphaTable(minSkinDensity, maxSkinDensity);

		for (int i = 0; i < g_tets.size(); i++) {
			n = 0;
			for (int j = 0; j < 4; j++)airVertexIndices[j] = -1;
			airVertexCount = 0; //@공기 접촉 면적..? -> 코드 리펙토링 필요
			skinVertexCount = 0;

			int i1 = g_tetIndices[i * 4 + 0];
			int i2 = g_tetIndices[i * 4 + 1];
			int i3 = g_tetIndices[i * 4 + 2];
			int i4 = g_tetIndices[i * 4 + 3];
			////sozipIncision
			//if ((vertices[i1].px == incisionIndex_1) && vertices[i1].py <= incisionIndex_2)continue; //incision
			//if ((vertices[i2].px == incisionIndex_1) && vertices[i2].py <= incisionIndex_2)continue;
			//if ((vertices[i3].px == incisionIndex_1) && vertices[i3].py <= incisionIndex_2)continue;
			//if ((vertices[i4].px == incisionIndex_1) && vertices[i4].py <= incisionIndex_2)continue;

			int arr[4] = { i1,i2,i3,i4 };
			for (int j = 0; j < 4; j++) {
				//if (AlphaTable[int(vertices[arr[j]].d)] <= 0) { //밀도가 작으니 공기라고 치겠다
				//    //printf("findAir");
				//    airVertexCount += 1;
				//    airVertexIndices[n++] = arr[j];
				//}
				float alpha = vertices[arr[j]].d;
				//printf("%f", alpha);
				//alpha = 1.0f - pow(2.71828, -(alpha - 1.0f));
				if (alpha < 0.8) { //밀도가 작으니 공기라고 치겠다
					//printf("findAir");
					airVertexCount += 1;
					airVertexIndices[n++] = arr[j];
				}
			}
			if (airVertexCount == 1) {
				//printf("findSkinTet\n"); 
				for (int j = 0; j < 4; j++) {
					//PBD밑작업

					if (arr[j] != airVertexIndices[0] &&
						//AlphaTable[int(vertices[arr[j]].d)] != 0 &&
						AlphaTable[int(vertices[arr[j]].d)] != 1 //이 부분 없으면 maxDensity 활용부분 모호
						) {
						// 피부로 판정되는 경우, skinVertices 배열에 저장
						skinVertices[skinVertexCount++] = arr[j];

						//printf("skinVertexCount : %d\n", skinVertices[0]);
						//printf("skinVertexCount : %d\n", skinVertices[1]);
						//printf("skinVertexCount : %d\n", skinVertices[2]);
					}
				}
				if (skinVertexCount == 3) {
					// skinVertices 배열에 저장된 정점들을 InitSkinSurface 함수에 전달
					//printf("findSkin\n");
					//printf("skinVertexCount : %d\n", skinVertices[0]);
					//printf("skinVertexCount : %d\n", skinVertices[1]);
					//printf("skinVertexCount : %d\n", skinVertices[2]);
					InitSkinSurface(skinVertices[0], skinVertices[1], skinVertices[2]);
				}
				else continue;
			}
			else continue;
		}
	}

	void Testbed::InitEdge() {

		int nt = (PBD_X - 1) * (PBD_Y - 1) * (PBD_Z - 1) * 5;


		typedef std::vector<unsigned int> VertexEdges;
		int nv = PBD_X * PBD_Y * PBD_Z;

		g_verticesEdges.resize(nv);

		g_edges.clear();

		g_tets.resize(nt);

		for (unsigned int i = 0u; i < indices.size(); i++) {
			g_tetIndices.push_back(indices[i]);
			//printf("m_tetIdices  %d\n", g_tetIndices[i]);
		}

		g_tetIndices.resize(nt * 4);

		for (unsigned int i = 0; i < nt; i++)
		{
			// tet edge indices: {0,1, 0,2, 0,3, 1,2, 1,3, 2,3}
			//g_tetIndices => simCell을 이루는 tetra 각각의 좌표
			const unsigned int edges[12] = { g_tetIndices[4 * i], g_tetIndices[4 * i + 1],
				g_tetIndices[4 * i], g_tetIndices[4 * i + 2],
				g_tetIndices[4 * i], g_tetIndices[4 * i + 3],
				g_tetIndices[4 * i + 1], g_tetIndices[4 * i + 2],
				g_tetIndices[4 * i + 1], g_tetIndices[4 * i + 3],
				g_tetIndices[4 * i + 2], g_tetIndices[4 * i + 3] };



			// tet face indices: {1,0,2, 3,1,2, 0,3,2, 0,1,3} => counter clock wise
			const unsigned int faces[12] = { g_tetIndices[4 * i + 1], g_tetIndices[4 * i], g_tetIndices[4 * i + 2],
				g_tetIndices[4 * i + 3], g_tetIndices[4 * i + 1], g_tetIndices[4 * i + 2],
				g_tetIndices[4 * i], g_tetIndices[4 * i + 3], g_tetIndices[4 * i + 2],
				g_tetIndices[4 * i], g_tetIndices[4 * i + 1], g_tetIndices[4 * i + 3] };

			//for (unsigned int j = 0u; j < 4; j++)
			//{
			//   // add vertex-tet connection
			//   const unsigned int vIndex = g_tetIndices[4 * i + j];
			//   m_verticesTets[vIndex].push_back(i);
			//}

			//for (unsigned int j = 0u; j < 4; j++)
			//{
			//    // add face information
			//    const unsigned int a = faces[j * 3 + 0];
			//    const unsigned int b = faces[j * 3 + 1];
			//    const unsigned int c = faces[j * 3 + 2];
			//    unsigned int face = 0xffffffff;
			//    // find face
			//    for (unsigned int k = 0; k < m_verticesFaces[a].size(); k++)
			//    {
			//        // Check if we already have this face in the list
			//        const unsigned int& faceIndex = m_verticesFaces[a][k];
			//        if (((g_faceIndices[3 * faceIndex] == a) || (g_faceIndices[3 * faceIndex] == b) || (g_faceIndices[3 * faceIndex] == c)) &&
			//            ((g_faceIndices[3 * faceIndex + 1] == a) || (g_faceIndices[3 * faceIndex + 1] == b) || (g_faceIndices[3 * faceIndex + 1] == c)) &&
			//            ((g_faceIndices[3 * faceIndex + 2] == a) || (g_faceIndices[3 * faceIndex + 2] == b) || (g_faceIndices[3 * faceIndex + 2] == c)))
			//        {
			//            face = m_verticesFaces[a][k];
			//            break;
			//        }
			//    }
			//    if (face == 0xffffffff)
			//    {
			//        // create new
			//        Face f;
			//        g_faceIndices.push_back(a);
			//        g_faceIndices.push_back(b);
			//        g_faceIndices.push_back(c);
			//        face = (unsigned int)g_faceIndices.size() / 3 - 1u;
			//        f.m_tets[0] = i;
			//        f.m_tets[1] = 0xffffffff;
			//        m_faces.push_back(f);
			//
			//        // add vertex-face connection            
			//        m_verticesFaces[a].push_back(face);
			//        m_verticesFaces[b].push_back(face);
			//        m_verticesFaces[c].push_back(face);
			//    }
			//    else
			//    {
			//        Face& fd = m_faces[face];
			//        fd.m_tets[1] = i;
			//    }
			//    // append face
			//    m_tets[i].m_faces[j] = face;
			//}

			for (unsigned int j = 0u; j < 6; j++)
			{
				// add face information
				const unsigned int a = edges[j * 2 + 0];
				const unsigned int b = edges[j * 2 + 1];
				unsigned int edge = 0xffffffff;
				// find edge
				for (unsigned int k = 0; k < g_verticesEdges[a].size(); k++)
				{
					// Check if we already have this edge in the list
					const Edge& e = g_edges[g_verticesEdges[a][k]];
					if (((e.m_vert[0] == a) || (e.m_vert[0] == b)) &&
						((e.m_vert[1] == a) || (e.m_vert[1] == b)))
					{
						edge = g_verticesEdges[a][k];
						break;
					}
				}
				if (edge == 0xffffffff)
				{
					// create new
					Edge e;
					e.m_vert[0] = a;
					e.m_vert[1] = b;

					//if ((vertices[a].px == incisionIndex_1) && vertices[a].py < incisionIndex_2)continue; //incision
					//if ((vertices[b].px == incisionIndex_1) && vertices[b].py < incisionIndex_2)continue;
					g_edges.push_back(e);

					edge = (unsigned int)g_edges.size() - 1u;

					// add vertex-edge connection            
					g_verticesEdges[a].push_back(edge);
					g_verticesEdges[b].push_back(edge);
				}
				// append edge
				g_tets[i].m_edges[j] = edge;
			}
		}
		//get Stiffness
		/**  new Algorithm  **/
		//DetectSurface();
		//get RestLength

		//get RestLength
		for (int i = 0; i < g_edges.size(); i++) {
			int i1 = g_edges[i].m_vert[0];
			int i2 = g_edges[i].m_vert[1];
			//if ((vertices[i1].px == incisionIndex_1) && vertices[i1].py < incisionIndex_2)continue; //incision
			//if ((vertices[i2].px == incisionIndex_1) && vertices[i2].py < incisionIndex_2)continue;
			float p1[3] = { vertices[i1].px, vertices[i1].py, vertices[i1].pz };
			float p2[3] = { vertices[i2].px, vertices[i2].py, vertices[i2].pz };
			float rl = GetLength(p1, p2);

			if (g_edges[i].rl == 0)
				g_edges[i].rl = rl;
			//if (vertices[g_edges[i].m_vert[0]].pz == 0) g_edges[i].rl = rl * LengthShrink;
			float avg_density = (vertices[i1].d + vertices[i2].d) / 2;
			float stiffness = 0.5;//avg_density / 1000.;

			g_edges[i].st = stiffness;
			distance_constraints.push_back(DistanceConstraint(i1, i2, rl, g_edges[i].st));
			//initConstraint(i1, i2, rl, g_edges[i].st);
			//m_
		}
		edges = (Edge*)malloc(sizeof(Edge) * g_edges.size()); //edges for cuda
		for (int i = 0; i < g_edges.size(); i++) {
			edges[i] = g_edges[i];
		}

		DetectAndInitSkinSurface();
		printf("skinAlgorithm\n");
		// for(int i =0; i<)
		for (int i = 0; i < g_skins.size(); i++) {

			int i1 = g_skins[i].m_vert[0];
			int i2 = g_skins[i].m_vert[1];
			int i3 = g_skins[i].m_vert[2];

			// 경도 초기화
			float p1[3] = { vertices[i1].px, vertices[i1].py, vertices[i1].pz };
			float p2[3] = { vertices[i2].px, vertices[i2].py, vertices[i2].pz };
			float p3[3] = { vertices[i3].px, vertices[i3].py, vertices[i3].pz };

			//if ((vertices[i1].px == incisionIndex_1) && vertices[i1].py < incisionIndex_2)continue; //incision
			//if ((vertices[i2].px == incisionIndex_1) && vertices[i2].py < incisionIndex_2)continue;
			//if ((vertices[i3].px == incisionIndex_1) && vertices[i3].py < incisionIndex_2)continue;

			float rsa = TriangleArea3D(p1, p2, p3);
			if (g_skins[i].rsa == 0)
				g_skins[i].rsa = rsa;
			if (vertices[g_skins[i].m_vert[0]].pz == 0) g_skins[i].rsa = rsa * 0.5;
			float avg_density = (vertices[i1].d + vertices[i2].d + vertices[i3].d) / 3;
			//초기화,,?
			//float stiffness = avg_density / 500;

			g_skins[i].st = 0.9;//stiffness;

			//printf("%d\n", i);
			skin_constraints.push_back(SkinTenstionConstraint(i1, i2, i3, g_skins[i].rsa, g_skins[i].st));
		}

		skins = (SkinSurface*)malloc(sizeof(SkinSurface) * g_skins.size()); //skins for cuda
		for (int i = 0; i < g_skins.size(); i++) {
			skins[i] = g_skins[i];
		}
		//get RestVolume
		for (int i = 0; i < g_tets.size(); i++) { //i = 0 >> 0123, i=4 4567
			int i1 = g_tetIndices[i * 4 + 0];
			int i2 = g_tetIndices[i * 4 + 1];
			int i3 = g_tetIndices[i * 4 + 2];
			int i4 = g_tetIndices[i * 4 + 3];
			//if ((vertices[i1].px == incisionIndex_1) && vertices[i1].py < incisionIndex_2)continue;//incision
			//if ((vertices[i2].px == incisionIndex_1) && vertices[i2].py < incisionIndex_2)continue;
			//if ((vertices[i3].px == incisionIndex_1) && vertices[i3].py < incisionIndex_2)continue;
			//if ((vertices[i4].px == incisionIndex_1) && vertices[i4].py < incisionIndex_2)continue;
			g_tets[i].m_vert[0] = i1;
			g_tets[i].m_vert[1] = i2;
			g_tets[i].m_vert[2] = i3;
			g_tets[i].m_vert[3] = i4;
			float p0[3] = { vertices[i1].px, vertices[i1].py, vertices[i1].pz };
			float p1[3] = { vertices[i2].px, vertices[i2].py, vertices[i2].pz };
			float p2[3] = { vertices[i3].px, vertices[i3].py, vertices[i3].pz };
			float p3[3] = { vertices[i4].px, vertices[i4].py, vertices[i4].pz };
			// tetra Volume = 1./6 * (a x b) * c 
			float avector[3], bvector[3], cvector[3];
			float avg_density = (vertices[i1].d + vertices[i2].d + vertices[i3].d + vertices[i4].d) / 4;
			float stiffness = 0.5;//avg_density / 1000;
			//if (stiffness > 2)stiffness = 2;
			//else stiffness += 0.2;
			g_tets[i].st = stiffness;
			avector[0] = makeVector(p1, p0)[0];
			avector[1] = makeVector(p1, p0)[1];
			avector[2] = makeVector(p1, p0)[2];

			bvector[0] = makeVector(p2, p0)[0];
			bvector[1] = makeVector(p2, p0)[1];
			bvector[2] = makeVector(p2, p0)[2];

			cvector[0] = makeVector(p3, p0)[0];
			cvector[1] = makeVector(p3, p0)[1];
			cvector[2] = makeVector(p3, p0)[2];

			float crossResult[3];
			crossResult[0] = crossProduct(avector, bvector)[0];
			crossResult[1] = crossProduct(avector, bvector)[1];
			crossResult[2] = crossProduct(avector, bvector)[2];
			//crossResult[1] = { crossProduct(avector, bvector)[1] };
			//crossResult[2] = { crossProduct(avector, bvector)[2] };
			float volume = fabsf((1.0 / 6.0) * dotProduct(crossResult, cvector));
			g_tets[i].rv = volume;
			volume_constraints.push_back(VolumeConstraint(i1, i2, i3, i4, g_tets[i].st, g_tets[i].rv));
			//float p0[3] = { vertices[i1].px, vertices[i1].py, vertices[i1].pz };
			//float p1[3] = { vertices[i2].px, vertices[i2].py, vertices[i2].pz };
			//float p2[3] = { vertices[i3].px, vertices[i3].py, vertices[i3].pz };
			//float p3[3] = { vertices[i4].px, vertices[i4].py, vertices[i4].pz };

			float e[3] = { p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2] };
			float  elen = GetLength(p3, p2);
			if (elen < 1e-6)
				return;

			float invElen = static_cast<float>(1.0) / elen;

			float n1[3], n2[3];
			//float avector[3], bvector[3], cvector[3];
			//Vector3r grad1 = (p2 - p0).cross(p3 - p0);
			//makeVector(p0, p2, avector);
			avector[0] = makeVector(p0, p2)[0];
			avector[1] = makeVector(p0, p2)[1];
			avector[2] = makeVector(p0, p2)[2];
			//makeVector(p0, p3, bvector);
			bvector[0] = makeVector(p0, p3)[0];
			bvector[1] = makeVector(p0, p3)[1];
			bvector[2] = makeVector(p0, p3)[2];
			//crossProduct(avector, bvector, grad1);
			n1[0] = crossProduct(avector, bvector)[0];
			n1[1] = crossProduct(avector, bvector)[1];
			n1[2] = crossProduct(avector, bvector)[2];

			n1[0] /= squaredNorm(n1);
			n1[1] /= squaredNorm(n1);
			n1[2] /= squaredNorm(n1);
			/// <summary>
			/// ////////////////////////////////////////////////////////////////////////////////////////////
			/// </summary>
			avector[0] = makeVector(p1, p3)[0];
			avector[1] = makeVector(p1, p3)[1];
			avector[2] = makeVector(p1, p3)[2];
			//makeVector(p0, p3, bvector);
			bvector[0] = makeVector(p1, p2)[0];
			bvector[1] = makeVector(p1, p2)[1];
			bvector[2] = makeVector(p1, p2)[2];
			//crossProduct(avector, bvector, grad1);
			n2[0] = crossProduct(avector, bvector)[0];
			n2[1] = crossProduct(avector, bvector)[1];
			n2[2] = crossProduct(avector, bvector)[2];

			n2[0] /= squaredNorm(n2);
			n2[1] /= squaredNorm(n2);
			n2[2] /= squaredNorm(n2);

			n1[0] = normalizeF(n1)[0];
			n1[1] = normalizeF(n1)[1];
			n1[2] = normalizeF(n1)[2];

			n2[0] = normalizeF(n2)[0];
			n2[1] = normalizeF(n2)[1];
			n2[2] = normalizeF(n2)[2];

			float dot = dotProduct(n1, n2);
			if (dot < -1.0) dot = -1.0;
			if (dot > 1.0) dot = 1.0;
			float ra = acos(dot);
			g_tets[i].ra = ra;
			//dihedral_constraints.push_back(DihedralConstraint(i1, i2, i3, i4, g_tets[i].st, g_tets[i].ra));
			/*



			float  elen = e.norm();
			if (elen < 1e-6)
				return false;

		float invElen = static_cast<Real>(1.0) / elen;

		//Vector3r grad1 = (p2 - p0).cross(p3 - p0);
		makeVector(p0, p2, avector);
		//avector[0] = makeVector(p0, p2)[0];
		//avector[1] = makeVector(p0, p2)[1];
		//avector[2] = makeVector(p0, p2)[2];
		makeVector(p0, p3, bvector);
		//bvector[0] = makeVector(p0, p3)[0];
		//bvector[1] = makeVector(p0, p3)[1];
		//bvector[2] = makeVector(p0, p3)[2];
		crossProduct(avector, bvector, grad1);
		//grad1[0] = crossProduct(avector, bvector)[0];
		//grad1[1] = crossProduct(avector, bvector)[1];
		//grad1[2] = crossProduct(avector, bvector)[2];

		Vector3r n1 = (p2 - p0).cross(p3 - p0);
		n1 /= n1.squaredNorm();
		Vector3r n2 = (p3 - p1).cross(p2 - p1);
		n2 /= n2.squaredNorm();

		n1.normalize();
		n2.normalize();

		float dot = n1.dot(n2);

		if (dot < -1.0) dot = -1.0;
		if (dot > 1.0) dot = 1.0;

		float ra = acos(dot);
		g_tets[i].ra=ra;
		  dihedral_constraints.push_back(DihedralConstraint(i1, i2, i3, i4, g_tets[i].st, g_tets[i].ra));
			*/

		}
		tets = (Tet*)malloc(sizeof(Tet) * g_tets.size()); //edges for cuda
		for (int i = 0; i < g_tets.size(); i++) {
			tets[i] = g_tets[i];
		}


	}

	void Testbed::InitVertices() { //좌표 초기화
		//for... {
		//   vertices[]
		//}
		//z* VOLY* VOLX + y * VOLX + x;
		const int X = PBD_X;
		const int Y = PBD_Y;
		const int Z = PBD_Z;
		//34*16 =
		//for (int i = 0; i < VOLX - 1; i++)
		//{
		//   for (int j = 0; j < VOLY - 1; j++)
		//   {
		//      for (int k = 0; k < VOLZ - 1; k++)
		//i * VOLY * VOLZ + j * VOLZ + k

		for (int z = 0; z < Z; z++) {
			for (int y = 0; y < Y; y++) {
				for (int x = 0; x < X; x++) {
					float z_temp = z;
					float y_temp = y;
					float x_temp = x;
					int index = x * Y * Z + y * Z + z;// z * Y * X + y * X + x;
					int linearIndex = 0;
					vertices[index].px = x_temp;
					vertices[index].py = y_temp;
					vertices[index].pz = z_temp;

					//if ((vertices[index].px == incisionIndex_1) && vertices[index].py < incisionIndex_2)continue; //incision
					//if ((vertices[index].px == incisionIndex_1 || vertices[index].px == incisionIndex_1_1) && vertices[index].py < incisionIndex_2) {
					//    vertices[index].vx = 0;
					//    vertices[index].vy = 0;
					//    vertices[index].vz = 0;
					//    vertices[index].d = 0;
					//    vertices[index].ax = 0;
					//    vertices[index].ay = 0;
					//    vertices[index].az = 0;
					//    vertices[index].m = 0;
					//    vertices[index].im = 0;
					//    continue; //O
					//}
					vertices[index].vx = 0;
					vertices[index].vy = 0;
					vertices[index].vz = 0;
					/*init acceleration*/
					vertices[index].ax = 0;
					vertices[index].ay = 0;
					vertices[index].az = 0;

					/* init density */
					//if ((PBD_SCALE * z_temp < 512) && (y_temp * PBD_SCALE < 512) && (x_temp * PBD_SCALE < 512))
					//    vertices[index].d = vol[(int)z_temp * (PBD_SCALE - 2)][(int)y_temp * PBD_SCALE][(int)x_temp * PBD_SCALE];
					//printf("%d : %7.1f  ",index, vertices[index].d);
					//PBD밑작업
					if ((PBD_SCALE * z_temp < 256) && (y_temp * PBD_SCALE < 256) && (x_temp * PBD_SCALE < 256))
						//여기 바꿔야함
						linearIndex = (int)x_temp * PBD_SCALE * VOLX * VOLY + (int)y_temp * PBD_SCALE * VOLX + (int)z_temp * PBD_SCALE;
					vertices[index].d = volume[linearIndex].w;
					//printf("Vertices : %f,Volume : %f\n", vertices[index].d, volume[linearIndex].w);
				//printf("volume Test : %f, %f\n", vertices[index].d, volume[index].w);

					if (vertices[index].px == PBD_X || vertices[index].px == 0 ||
						vertices[index].py == PBD_Y || vertices[index].py == 0 ||
						vertices[index].pz == PBD_Z || vertices[index].pz == 0
						) {

						vertices[index].im = 0.0; //infinite mass
						vertices[index].az = 0; // 벽에 붙은 친구들은 힘을 받지 않는다.
						//vertices[z * Y * Z + y * Z + x].m = 0.0
					}
					else {
						vertices[index].m = 1;
						vertices[index].im = 1;
						//if (vertices[index].d < 300) {
						//    vertices[index].m = 0.1;
						//    vertices[index].im = 10;
						//}
					}
					//printf("%f %f %f\n", vertices[z * Y * Z + y * Z + x].px, vertices[z * Y * Z + y * Z + x].py, vertices[z * Y * Z + y * Z + x].pz);
				}
				//printf("\n");
			}
			//printf("\n");
		}




		indices.reserve(PBD_X * PBD_Y * PBD_Z * 5);
		for (int i = 0; i < PBD_X - 1; i++)
		{
			for (int j = 0; j < PBD_Y - 1; j++)
			{
				for (int k = 0; k < PBD_Z - 1; k++)
				{

					unsigned int p0 = i * PBD_Y * PBD_Z + j * PBD_Z + k;
					unsigned int p1 = p0 + 1;
					unsigned int p3 = (i + 1) * PBD_Y * PBD_Z + j * PBD_Z + k;
					unsigned int p2 = p3 + 1;
					unsigned int p7 = (i + 1) * PBD_Y * PBD_Z + (j + 1) * PBD_Z + k;
					unsigned int p6 = p7 + 1;
					unsigned int p4 = i * PBD_Y * PBD_Z + (j + 1) * PBD_Z + k;
					unsigned int p5 = p4 + 1;


					if ((i + j + k) % 2 == 1)
					{
						indices.push_back(p2); indices.push_back(p1); indices.push_back(p6); indices.push_back(p3);
						indices.push_back(p6); indices.push_back(p3); indices.push_back(p4); indices.push_back(p7);
						indices.push_back(p4); indices.push_back(p1); indices.push_back(p6); indices.push_back(p5);
						indices.push_back(p3); indices.push_back(p1); indices.push_back(p4); indices.push_back(p0);
						indices.push_back(p6); indices.push_back(p1); indices.push_back(p4); indices.push_back(p3);
					}
					else
					{
						indices.push_back(p0); indices.push_back(p2); indices.push_back(p5); indices.push_back(p1);
						indices.push_back(p7); indices.push_back(p2); indices.push_back(p0); indices.push_back(p3);
						indices.push_back(p5); indices.push_back(p2); indices.push_back(p7); indices.push_back(p6);
						indices.push_back(p7); indices.push_back(p0); indices.push_back(p5); indices.push_back(p4);
						indices.push_back(p0); indices.push_back(p2); indices.push_back(p7); indices.push_back(p5);
					}

				}
			}
		}
	}

	void Testbed::Update() { //위치가 변함을 표현 즉, force작용  


		for (int i = 0; i < PBD_X * PBD_Y * PBD_Z; i++) { // 
			Vertex& t = vertices[i];

			//if (t.py == 9.0) {
			//    t.im = 0.0; //infinite mass
			//    t.ay = 0; // 벽에 붙은 친f구들은 중력의 힘을 받지 않는다.
			//                            //vertices[z * Y * Z + y * Z + x].m = 0.0
			//}
			if (t.px == PBD_X || t.px == 0 ||
				t.py == PBD_Y || t.py == 0 ||
				t.pz == PBD_Z || t.pz == 0
				) {

				t.im = 0.0; //infinite mass
				t.az = 0; // 벽에 붙은 친구들은 힘을 받지 않는다.
				//vertices[z * Y * Z + y * Z + x].m = 0.0
			}
			t.vx = t.vx + t.ax * dt;
			t.vy = t.vy + t.ay * dt;
			t.vz = t.vz + t.az * dt;
			vertices[i].vx = t.vx;
			vertices[i].vy = t.vy;

			vertices[i].vz = t.vz;

			t.px = t.px + t.vx * dt;
			t.py = t.py + t.vy * dt;
			t.pz = t.pz + t.vz * dt;
			vertices[i].px = t.px;
			vertices[i].py = t.py;
			vertices[i].pz = t.pz;

			// dx,dy,dz를 0으로 초기화
			vertices[i].dx = 0;
			vertices[i].dy = 0;
			vertices[i].dz = 0;
			//
			//Vertex t = vertices[i]; 
			//t.vx = t.vx + t.ax * dt; 
			//t.vy = t.vy + t.ay * dt; 
			//t.vz = t.vz + t.az * dt; 
			//vertices[i].vx = t.vx;
			//vertices[i].vy = t.vy;
			//vertices[i].vz = t.vz;
			//
			//
			//t.px = t.px + t.vx * dt; //
			//t.py = t.py + t.vy * dt; //
			//t.pz = t.pz + t.vz * dt; //
			//vertices[i].px = t.px;
			//vertices[i].py = t.py;
			//vertices[i].pz = t.pz;s
			//
			//// DONE: dx,dy,dz를 0으로 초기화
			//vertices[i].dx = 0;
			//vertices[i].dy = 0;
			//vertices[i].dz = 0;
		}

	}

	void Testbed::velocityUpdate() {
		// TODO: 각 vertex에 대해서 속도 재조정
		// 함수 하나 만들어서
		// vertices[i].vx += vertices[i].dx / dt;
		int size = PBD_X * PBD_Y * PBD_Z;
		for (int i = 0; i < size; i++) {
			vertices[i].vx += vertices[i].dx / dt;
			vertices[i].vy += vertices[i].dy / dt;
			vertices[i].vz += vertices[i].dz / dt;
		}
	}

	void Testbed::InitParallelizableGroups() {
		if (m_groupsInitialized)
			return;
		std::cout << "InitParallelizableGroups Func" << '\n';
		const unsigned int numDistanceConstraints = (unsigned int)g_edges.size(); // (unsigned int)m_tets.size() 
		const unsigned int numVolumeConstraints = (unsigned int)g_tets.size();
		const unsigned int numSkinConstraints = (unsigned int)g_skins.size();//thth
		const unsigned int numParticles = (unsigned int)PBD_X * PBD_Y * PBD_Z;
		//const unsigned int numRigidBodies = (unsigned int)m_rigidBodies.size();
		const unsigned int numBodies = numParticles;// +numRigidBodies;
		g_prallelizableEdgeGroups.clear();
		g_prallelizableTetGroups.clear();
		g_prallelizableSkinGroups.clear(); //thth

		// Maps in which group a particle is or 0 if not yet mapped
		std::vector<unsigned char*> mapping;
		//에지개수만큼 반복 -> 적용해야하는 Constraint수만큼 반복 즉, numEdges+numTets
		for (unsigned int i = 0; i < numDistanceConstraints; i++)
		{
			//처리해야하는 에지번호
			Constraint constraint = distance_constraints[i];

			bool addToNewGroup = true;
			//기존 그룹 개수만큼 돌면서
			for (unsigned int j = 0; j < g_prallelizableEdgeGroups.size(); j++)
			{
				bool addToThisGroup = true;
				//numOfbod>아마 정점개수일것 (2개) >> edge일땐, 2번 tets일땐 4번
				for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
				{
					//이미 정점이 있어?
					if (mapping[j][constraint.m_bodies[k]] != 0)
					{

						//정점이 있으면, 이 그룹에서는 받을수가없어
						addToThisGroup = false;
						break;
					}
				}
				//이 그룹에서 받을수 있으면
				if (addToThisGroup)
				{
					g_prallelizableEdgeGroups[j].push_back(i);
					//넣고, 등록한다.
					for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
						mapping[j][constraint.m_bodies[k]] = 1;
					//받았으니, 새로 그룹을 만들 필요는 없다 
					addToNewGroup = false;
					break;
				}
			}
			//새로 그룹을 만들 필요가 있는경우
			if (addToNewGroup)
			{
				mapping.push_back(new unsigned char[numBodies]);
				memset(mapping[mapping.size() - 1], 0, sizeof(unsigned char) * numBodies);
				g_prallelizableEdgeGroups.resize(g_prallelizableEdgeGroups.size() + 1);
				g_prallelizableEdgeGroups[g_prallelizableEdgeGroups.size() - 1].push_back(i);
				for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
					mapping[g_prallelizableEdgeGroups.size() - 1][constraint.m_bodies[k]] = 1;
			}
		}
		//왜 여기는 clear고  volume은 배열로 돌면서 clear치나요? - thth
		mapping.clear();

		std::cout << "Volume number" << numVolumeConstraints << '\n';

		for (unsigned int i = 0; i < numVolumeConstraints; i++)
		{
			//처리해야하는 에지번호
			Constraint constraint = volume_constraints[i];

			bool addToNewGroup = true;
			//기존 그룹 개수만큼 돌면서
			for (unsigned int j = 0; j < g_prallelizableTetGroups.size(); j++)
			{
				bool addToThisGroup = true;
				//numOfbod>아마 정점개수일것 (2개) >> edge일땐, 2번 tets일땐 4번
				for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
				{
					//이미 정점이 있어?
					if (mapping[j][constraint.m_bodies[k]] != 0)
					{

						//정점이 있으면, 이 그룹에서는 받을수가없어
						addToThisGroup = false;
						break;
					}
				}
				//이 그룹에서 받을수 있으면
				if (addToThisGroup)
				{
					g_prallelizableTetGroups[j].push_back(i);
					//넣고, 등록한다.
					for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
						mapping[j][constraint.m_bodies[k]] = 1;
					//받았으니, 새로 그룹을 만들 필요는 없다 
					addToNewGroup = false;
					break;
				}
			}
			//새로 그룹을 만들 필요가 있는경우
			if (addToNewGroup)
			{
				mapping.push_back(new unsigned char[numBodies]);
				memset(mapping[mapping.size() - 1], 0, sizeof(unsigned char) * numBodies);
				g_prallelizableTetGroups.resize(g_prallelizableTetGroups.size() + 1);
				g_prallelizableTetGroups[g_prallelizableTetGroups.size() - 1].push_back(i);
				for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
					mapping[g_prallelizableTetGroups.size() - 1][constraint.m_bodies[k]] = 1;
			}
		}
		for (unsigned int i = 0; i < mapping.size(); i++)
		{
			delete[] mapping[i];
		}
		mapping.clear();

		std::cout << "Skin number" << numSkinConstraints << '\n';
		//스킨 그룹 - thth
		for (unsigned int i = 0; i < numSkinConstraints; i++)
		{
			//처리해야하는 에지번호
			//cout << "@@@@@" << (int)g_prallelizableSkinGroups.size() << endl;
			Constraint constraint = skin_constraints[i];
			bool addToNewGroup = true;
			//기존 그룹 개수만큼 돌면서
			for (unsigned int j = 0; j < g_prallelizableSkinGroups.size(); j++)
			{
				bool addToThisGroup = true;
				//numOfbod>아마 정점개수일것 (2개) >> edge일땐, 2번 tets일땐 4번 skin일때는 3번
				for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
				{
					//이미 정점이 있어?
					if (mapping[j][constraint.m_bodies[k]] != 0)
					{

						//정점이 있으면, 이 그룹에서는 받을수가없어
						addToThisGroup = false;
						break;
					}
				}
				//이 그룹에서 받을수 있으면
				if (addToThisGroup)
				{
					g_prallelizableSkinGroups[j].push_back(i);
					//넣고, 등록한다.
					for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
						mapping[j][constraint.m_bodies[k]] = 1;
					//받았으니, 새로 그룹을 만들 필요는 없다 
					addToNewGroup = false;
					break;
				}
			}
			//새로 그룹을 만들 필요가 있는경우
			if (addToNewGroup)
			{
				mapping.push_back(new unsigned char[numBodies]);
				memset(mapping[mapping.size() - 1], 0, sizeof(unsigned char) * numBodies);
				g_prallelizableSkinGroups.resize(g_prallelizableSkinGroups.size() + 1);
				g_prallelizableSkinGroups[g_prallelizableSkinGroups.size() - 1].push_back(i);
				for (unsigned int k = 0; k < constraint.numberOfBodies(); k++)
					mapping[g_prallelizableSkinGroups.size() - 1][constraint.m_bodies[k]] = 1;
			}
		}
		for (unsigned int i = 0; i < mapping.size(); i++)
		{
			delete[] mapping[i];
		}
		mapping.clear();
		//for (int a = 0; a < g_prallelizableSkinGroups.size(); a++) {
		//    cout << a << " :";
		//    for (int b = 0; b < g_prallelizableSkinGroups[a].size(); b++) {
		//        cout << g_prallelizableSkinGroups[a][b]<<" ";
		//    }
		//    cout << endl;
		//}
		m_groupsInitialized = true;

	}

	void Testbed::Projection() {
		//수정필요
		//#pragma omp parallel
		for (int i = 0; i < g_prallelizableEdgeGroups.size(); i++) {
			Testbed::cuda_SolveDistanceConstraint((int*)(g_prallelizableEdgeGroups[i].data()), g_prallelizableEdgeGroups[i].size());

		}
		//#pragma omp parallel
		for (int i = 0; i < g_prallelizableTetGroups.size(); i++) {
			Testbed::cuda_SolveVolumeConstraint((int*)(g_prallelizableTetGroups[i].data()), g_prallelizableTetGroups[i].size());

		}
		//#pragma omp parallel
			//for (int i = 0; i < g_prallelizableTetGroups.size(); i++) {
			//   Testbed::cuda_SolveDihedralConstraint((int*)(g_prallelizableTetGroups[i].data()), g_prallelizableTetGroups[i].size());
			//}

		 //#pragma omp parallel

		for (int i = 0; i < g_prallelizableSkinGroups.size(); i++) {
			Testbed::cuda_SolveSkinTensionConstraint((int*)(g_prallelizableSkinGroups[i].data()), g_prallelizableSkinGroups[i].size());

		}

	}

	int Testbed::PBDGPUMemoryAlloc(int E_SIZE, int CG_SIZE, int T_SIZE, int T_CG_SIZE, int S_SIZE, int S_CG_SIZE) {
		//초기 한번만 해주면 되는 것들 Vertices, Edge에 대한 memory Allocation, memcpy
		printf("I'm in PBDGPUMemoryAlloc\n");
		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&d_DistanceGroup, CG_SIZE * sizeof(int)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMalloc failed1\n");
			return -1;
		}
		printf("1\n");

		cudaStatus = cudaMalloc((void**)&d_VolumeGroup, T_CG_SIZE * sizeof(int)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMalloc failed2\n");
			return -1;
		}

		cudaStatus = cudaMalloc((void**)&d_SkinGroup, S_CG_SIZE * sizeof(int)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMalloc failed3\n");
			return -1;
		}


		cudaStatus = cudaMalloc((void**)&d_vertices, PBD_X * PBD_Y * PBD_Z * sizeof(Vertex)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			printf("device Vertices cudaMalloc failed!\n");
			return cudaStatus;
		}
		printf("2\n");

		cudaStatus = cudaMemcpy(d_vertices, vertices, PBD_X * PBD_Y * PBD_Z * sizeof(Vertex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {

			printf("Device Vertices cudaMemcpy failed!\n");
			return cudaStatus;
		}
		printf("3\n");

		cudaStatus = cudaMalloc((void**)&d_edges, E_SIZE * sizeof(Edge)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Edges cudaMalloc failed!\n");
			return cudaStatus;
		}
		printf("4\n");


		cudaStatus = cudaMemcpy(d_edges, edges, E_SIZE * sizeof(Edge), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Edges cudaMemcpy failed!\n");
			return cudaStatus;
		}
		cudaStatus = cudaMalloc((void**)&d_tets, T_SIZE * sizeof(Tet)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Edges cudaMalloc failed!\n");
			return cudaStatus;
		}
		printf("5\n");


		cudaStatus = cudaMemcpy(d_tets, tets, T_SIZE * sizeof(Tet), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Edges cudaMemcpy failed!\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&d_skins, S_SIZE * sizeof(SkinSurface)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Skins cudaMalloc failed!\n");
			return cudaStatus;
		}

		printf("6\n");

		cudaStatus = cudaMemcpy(d_skins, skins, S_SIZE * sizeof(SkinSurface), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Skins cudaMemcpy failed!\n");
			return cudaStatus;
		}

		return 0;
	}

	int Testbed::GPUtoCP() {
		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpy(vertices, d_vertices, (PBD_X) * (PBD_Y) * (PBD_Z) * sizeof(Vertex), cudaMemcpyDeviceToHost); //gpu->cpu로 복사

		//for (int i = 0; i < 256; i++) {
		//	for (int j = 0; j < 256; j++) {
		//		for (int k = 0; k < 256; k++) {
		//			int idx = i * 256 * 256 + j * 256 + k;
		//			printf("%f  %f  %f\n", d_vertices[idx].px, d_vertices[idx].py, d_vertices[idx].pz);
		//		}
		//	}
		//}

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "GPU to CP Vertices cudaMemcpy failed\n   ");
			return -1;
		}
		else return 0;
	}

	int Testbed::CPtoGPU() {
		cudaError_t cudaStatus;

		cudaStatus = cudaMemcpy(d_vertices, vertices, (PBD_X) * (PBD_Y) * (PBD_Z) * sizeof(Vertex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed! here\n");
			return cudaStatus;
		}

		//static bool has = false;
		//if (has) {
		//	for (int i = 0; i < 256; i++) {
		//		for (int j = 0; j < 256; j++) {
		//			for (int k = 0; k < 256; k++) {
		//				int idx = i * 256 * 256 + j * 256 + k;
		//				printf("%f  %f  %f\n", vertices[idx].px, vertices[idx].py, vertices[idx].pz);
		//			}
		//		}
		//	}
		//	printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
		//}
		//has = true;
		return 0;
	}

	void Testbed::PBDGPUMemoryFree() {
		cudaFree(d_vertices);
		cudaFree(d_edges);
		cudaFree(Testbed::d_DistanceGroup);
		cudaFree(Testbed::d_SkinGroup);
	}

	__device__ bool solve_SkinTensionConstraint(
		float x1, float y1, float z1, float invMass1,
		float x2, float y2, float z2, float invMass2,
		float x3, float y3, float z3, float invMass3,
		const float restSkinArea,/*restSkinArea is not necessary*/
		const float stiffness,
		float& corr1_x, float& corr1_y, float& corr1_z,
		float& corr2_x, float& corr2_y, float& corr2_z,
		float& corr3_x, float& corr3_y, float& corr3_z)
	{
		//return false;
		float centroid_x = (x1 + x2 + x3) / 3.0f;
		float centroid_y = (y1 + y2 + y3) / 3.0f;
		float centroid_z = (z1 + z2 + z3) / 3.0f;
		// Calculate vectors from centroid to each vertex (�� ���������κ��� �߽ɱ����� ���� ���)
		float vec1_x = centroid_x - x1;
		float vec1_y = centroid_y - y1;
		float vec1_z = centroid_z - z1;
		float vec2_x = centroid_x - x2;
		float vec2_y = centroid_y - y2;
		float vec2_z = centroid_z - z2;
		float vec3_x = centroid_x - x3;
		float vec3_y = centroid_y - y3;
		float vec3_z = centroid_z - z3;
		// Calculate lengths of vectors (������ ���� ���)
		float length1 = sqrt(vec1_x * vec1_x + vec1_y * vec1_y + vec1_z * vec1_z);
		float length2 = sqrt(vec2_x * vec2_x + vec2_y * vec2_y + vec2_z * vec2_z);
		float length3 = sqrt(vec3_x * vec3_x + vec3_y * vec3_y + vec3_z * vec3_z);
		// Normalize vectors (���� ����ȭ)
		if (length1 == 0 || length2 == 0 || length3 == 0) {
			return false; // �и� 0�� �Ǵ� ��� ���� ó��
		}
		if (invMass1 == 0 || invMass2 == 0 || invMass3 == 0)return false;
		vec1_x /= length1;
		vec1_y /= length1;
		vec1_z /= length1;
		vec2_x /= length2;
		vec2_y /= length2;
		vec2_z /= length2;
		vec3_x /= length3;
		vec3_y /= length3;
		vec3_z /= length3;
		// Calculate corrective forces (���� �� ���)
		float force1_x = stiffness * vec1_x;
		float force1_y = stiffness * vec1_y;
		float force1_z = stiffness * vec1_z;
		float force2_x = stiffness * vec2_x;
		float force2_y = stiffness * vec2_y;
		float force2_z = stiffness * vec2_z;
		float force3_x = stiffness * vec3_x;
		float force3_y = stiffness * vec3_y;
		float force3_z = stiffness * vec3_z;
		// Apply corrective forces to each point (�� ������ ���� �� ����)
		corr1_x = force1_x * invMass1;
		corr1_y = force1_y * invMass1;
		//corr1_z = 0;
		corr2_x = force2_x * invMass2;
		corr2_y = force2_y * invMass2;
		//corr2_z = 0;
		corr3_x = force3_x * invMass3;
		corr3_y = force3_y * invMass3;
		//corr3_z = 0;
		return true;
	}

	__device__ bool Solve_DistanceConstraint(
		float x1, float y1, float z1, float invMass1,
		float x2, float y2, float z2, float invMass2,
		const float restLength,
		const float stiffness,
		float& corr1_x, float& corr1_y, float& corr1_z,
		float& corr2_x, float& corr2_y, float& corr2_z,
		int vertid1, int vertid2
	)

	{
		//if(vertid1 == 9283) {
		//    printf("solve_DistanceConstraint:%f %f %f\n",x1,y1,z1);
		//}
		//if(vertid2 == 9283) {
		//    printf("solve_DistanceConstraint:%f %f %f\n",x2,y2,z2);
		//}

		float wSum = invMass1 + invMass2;
		if (wSum == 0.0) return false;
		float nl_x = x2 - x1;
		float nl_y = y2 - y1;
		float nl_z = z2 - z1;
		//direction vector(nl_x, nl_y, nl_z)
		float norm = sqrt(nl_x * nl_x + nl_y * nl_y + nl_z * nl_z);


		if (norm == 0) {
			printf("!!");
			printf("1:%f %f %f %d\n", x1, y1, z1, vertid1);
			printf("2:%f %f %f %d\n", x2, y2, z2, vertid2);
			return false;
		}
		float normalx = nl_x / norm;
		float normaly = nl_y / norm;
		float normalz = nl_z / norm;

		//float normalization_x1 = x1 / norm;
		//float normalization_y1 = y1 / norm;
		//float normalization_z1 = z1 / norm;
		//
		//float normalization_x2 = x2 / norm;
		//float normalization_y2 = y2 / norm;
		//float normalization_z2 = z2 / norm;
		float corr_x;
		float corr_y;
		float corr_z;
		corr_x = stiffness * normalx * (norm - restLength) / wSum;
		corr_y = stiffness * normaly * (norm - restLength) / wSum;
		corr_z = stiffness * normalz * (norm - restLength) / wSum;

		//corr_x *= damper;
		//corr_y *= damper;
		//corr_z *= damper;

		corr1_x = invMass1 * corr_x;
		corr1_y = invMass1 * corr_y;
		corr1_z = invMass1 * corr_z;

		corr2_x = -invMass2 * corr_x;
		corr2_y = -invMass2 * corr_y;
		corr2_z = -invMass2 * corr_z;


		float TH = 2.0;
		//if (fabs(corr1_x) > TH || fabs(corr2_x) > TH) {
		//	printf("??");
		//}
		return true;

	}

	__device__ bool solve_VolumeConstraint(
		float x1, float y1, float z1, float invMass1,
		float x2, float y2, float z2, float invMass2,
		float x3, float y3, float z3, float invMass3,
		float x4, float y4, float z4, float invMass4,
		const float restVolume,
		const float stiffness,
		float& corr1_x, float& corr1_y, float& corr1_z,
		float& corr2_x, float& corr2_y, float& corr2_z,
		float& corr3_x, float& corr3_y, float& corr3_z,
		float& corr4_x, float& corr4_y, float& corr4_z
	)
	{
		float p0[3] = { x1,y1,z1 };
		float p1[3] = { x2,y2,z2 };
		float p2[3] = { x3,y3,z3 };
		float p3[3] = { x4,y4,z4 };

		//float volume = static_cast<float>(1.0 / 6.0) * (p1 - p0).cross(p2 - p0).dot(p3 - p0);
		float avector[3], bvector[3], cvector[3];

		makeVector(p1, p0, avector);
		//avector[1] = makeVector(p1, p0)[1];
		//avector[2] = makeVector(p1, p0)[2];

		makeVector(p2, p0, bvector);
		//bvector[1] = makeVector(p2, p0)[1];
		//bvector[2] = makeVector(p2, p0)[2];

		makeVector(p3, p0, cvector);
		//cvector[1] = makeVector(p3, p0)[1];
		//cvector[2] = makeVector(p3, p0)[2];

		float crossResult[3];


		crossProduct(avector, bvector, crossResult);
		//crossResult[0] = crossProduct(avector, bvector)[0];
		//crossResult[1] = crossProduct(avector, bvector)[1];
		//crossResult[2] = crossProduct(avector, bvector)[2];


		float volume = fabsf((1.0 / 6.0) * dotProduct(crossResult, cvector));

		corr1_x = 0; corr1_y = 0; corr1_z = 0;
		corr2_x = 0; corr2_y = 0; corr2_z = 0;
		corr3_x = 0; corr3_y = 0; corr3_z = 0;
		corr4_x = 0; corr4_y = 0; corr4_z = 0;


		//corr0.setZero(); corr1.setZero(); corr2.setZero(); corr3.setZero();

		if (stiffness == 0.0)
			return false;


		float grad0[3], grad1[3], grad2[3], grad3[3];

		// Vector3r grad0 = (p1 - p2).cross(p3 - p2);
		makeVector(p2, p1, avector);
		//avector[0] = makeVector(p2, p1)[0];
		//avector[1] = makeVector(p2, p1)[1];
		//avector[2] = makeVector(p2, p1)[2];

		makeVector(p2, p3, bvector);
		//bvector[0] = makeVector(p2, p3)[0];
		//bvector[1] = makeVector(p2, p3)[1];
		//bvector[2] = makeVector(p2, p3)[2];

		crossProduct(avector, bvector, grad0);
		//grad0[0] = crossProduct(avector, bvector)[0];
		//grad0[1] = crossProduct(avector, bvector)[1];
		//grad0[2] = crossProduct(avector, bvector)[2];


		//Vector3r grad1 = (p2 - p0).cross(p3 - p0);
		makeVector(p0, p2, avector);
		//avector[0] = makeVector(p0, p2)[0];
		//avector[1] = makeVector(p0, p2)[1];
		//avector[2] = makeVector(p0, p2)[2];
		makeVector(p0, p3, bvector);
		//bvector[0] = makeVector(p0, p3)[0];
		//bvector[1] = makeVector(p0, p3)[1];
		//bvector[2] = makeVector(p0, p3)[2];
		crossProduct(avector, bvector, grad1);
		//grad1[0] = crossProduct(avector, bvector)[0];
		//grad1[1] = crossProduct(avector, bvector)[1];
		//grad1[2] = crossProduct(avector, bvector)[2];


		//Vector3r grad2 = (p0 - p1).cross(p3 - p1);
		makeVector(p1, p0, avector);
		//avector[0] = makeVector(p1, p0)[0];
		//avector[1] = makeVector(p1, p0)[1];
		//avector[2] = makeVector(p1, p0)[2];
		makeVector(p1, p3, bvector);
		//bvector[0] = makeVector(p1, p3)[0];
		//bvector[1] = makeVector(p1, p3)[1];
		//bvector[2] = makeVector(p1, p3)[2];
		crossProduct(avector, bvector, grad2);
		//grad2[0] = crossProduct(avector, bvector)[0];
		//grad2[1] = crossProduct(avector, bvector)[1];
		//grad2[2] = crossProduct(avector, bvector)[2];

		//Vector3r grad3 = (p1 - p0).cross(p2 - p0);
		makeVector(p0, p1, avector);
		//avector[0] = makeVector(p0, p1)[0];
		//avector[1] = makeVector(p0, p1)[1];
		//avector[2] = makeVector(p0, p1)[2];
		makeVector(p2, p0, bvector);
		//bvector[0] = makeVector(p2, p0)[0];
		//bvector[1] = makeVector(p2, p0)[1];
		//bvector[2] = makeVector(p2, p0)[2];
		crossProduct(avector, bvector, grad3);
		//grad3[0] = crossProduct(avector, bvector)[0];
		//grad3[1] = crossProduct(avector, bvector)[1];
		//grad3[2] = crossProduct(avector, bvector)[2];

		//float lambda =
		//    invMass0 * grad0.squaredNorm() +
		//    invMass1 * grad1.squaredNorm() +
		//    invMass2 * grad2.squaredNorm() +
		//    invMass3 * grad3.squaredNorm();


		float lambda =
			squaredNorm(grad0) * invMass1 +
			squaredNorm(grad1) * invMass2 +
			squaredNorm(grad2) * invMass3 +
			squaredNorm(grad3) * invMass4;

		if (fabs(lambda) <= 0.0)
			return false;

		lambda = stiffness * (volume - restVolume) / lambda;

		//corr0 = -lambda * invMass0 * grad0;
		//corr1 = -lambda * invMass1 * grad1;
		//corr2 = -lambda * invMass2 * grad2;
		//corr3 = -lambda * invMass3 * grad3;

		corr1_x = -lambda * invMass1 * grad0[0];
		corr2_x = -lambda * invMass2 * grad1[0];
		corr3_x = -lambda * invMass3 * grad2[0];
		corr4_x = -lambda * invMass4 * grad3[0];

		corr1_y = -lambda * invMass1 * grad0[1];
		corr2_y = -lambda * invMass2 * grad1[1];
		corr3_y = -lambda * invMass3 * grad2[1];
		corr4_y = -lambda * invMass4 * grad3[1];

		corr1_z = -lambda * invMass1 * grad0[2];
		corr2_z = -lambda * invMass2 * grad1[2];
		corr3_z = -lambda * invMass3 * grad2[2];
		corr4_z = -lambda * invMass4 * grad3[2];

		return true;
	}

	__device__ bool solve_DihedralConstraint(
		float x1, float y1, float z1, float invMass1,
		float x2, float y2, float z2, float invMass2,
		float x3, float y3, float z3, float invMass3,
		float x4, float y4, float z4, float invMass4,
		const float restAngle,
		const float stiffness,
		float& corr1_x, float& corr1_y, float& corr1_z,
		float& corr2_x, float& corr2_y, float& corr2_z,
		float& corr3_x, float& corr3_y, float& corr3_z,
		float& corr4_x, float& corr4_y, float& corr4_z
	)
	{
		if (invMass1 == 0.0 && invMass2 == 0.0)
			return false;
		float p0[3] = { x1,y1,z1 };
		float p1[3] = { x2,y2,z2 };
		float p2[3] = { x3,y3,z3 };
		float p3[3] = { x4,y4,z4 };

		float e[3] = { p3[0] - p2[0],p3[1] - p2[1] ,p3[2] - p2[2] };

		float elen = GetLength(p3, p2);
		if (elen < eps)
			return false;
		float invElen = static_cast<float>(1.0) / elen;
		float avector[3], bvector[3], cvector[3], crossResult[3], n1[3], n2[3];
		//Vector3 n1 = (p2 - p0).cross(p3 - p0); n1 /= n1.squaredNorm();
		makeVector(p0, p2, avector);
		makeVector(p0, p3, bvector);
		crossProduct(avector, bvector, crossResult);

		n1[0] = crossResult[0] / squaredNorm(crossResult);
		n1[1] = crossResult[1] / squaredNorm(crossResult);
		n1[2] = crossResult[2] / squaredNorm(crossResult);
		//Vector3 n2 = (p3 - p1).cross(p2 - p1); n2 /= n2.squaredNorm();
		makeVector(p1, p3, avector);
		makeVector(p1, p2, bvector);
		crossProduct(avector, bvector, crossResult);

		n2[0] = crossResult[0] / squaredNorm(crossResult);
		n2[1] = crossResult[1] / squaredNorm(crossResult);
		n2[2] = crossResult[2] / squaredNorm(crossResult);


		float d0[3], d1[3], d2[3], d3[3];
		d0[0] = elen * n1[0];
		d0[1] = elen * n1[1];
		d0[2] = elen * n1[2];

		d1[0] = elen * n2[0];
		d1[1] = elen * n2[1];
		d1[2] = elen * n2[2];
		//(p0 - p3).dotProduct(e) * invElen * n1 + (p1 - p3).dot(e) * invElen * n2;
		makeVector(p3, p0, avector);
		makeVector(p3, p1, bvector);

		d2[0] = dotProduct(avector, e) * invElen * n1[0] + dotProduct(bvector, e) * invElen * n2[0];
		d2[1] = dotProduct(avector, e) * invElen * n1[1] + dotProduct(bvector, e) * invElen * n2[1];
		d2[2] = dotProduct(avector, e) * invElen * n1[2] + dotProduct(bvector, e) * invElen * n2[2];



		//Vector3r d3 = (p2 - p0).dotProduct(e) * invElen * n1 + (p2 - p1).dot(e) * invElen * n2;
		makeVector(p0, p2, avector);
		makeVector(p1, p2, bvector);

		d3[0] = dotProduct(avector, e) * invElen * n1[0] + dotProduct(bvector, e) * invElen * n2[0];
		d3[1] = dotProduct(avector, e) * invElen * n1[1] + dotProduct(bvector, e) * invElen * n2[1];
		d3[2] = dotProduct(avector, e) * invElen * n1[2] + dotProduct(bvector, e) * invElen * n2[2];

		normalize(n1);
		normalize(n2);
		//n2.normalize();
		float dot = dotProduct(n1, n2);

		if (dot < -1.0) dot = -1.0;
		if (dot > 1.0) dot = 1.0;
		float phi = acos(dot);
		float lambda =
			invMass1 * squaredNorm(d0) +
			invMass2 * squaredNorm(d1) +
			invMass3 * squaredNorm(d2) +
			invMass4 * squaredNorm(d3);

		if (lambda == 0.0)
			return false;

		lambda = (phi - restAngle) / lambda * stiffness;
		crossProduct(n1, n2, crossResult);
		if (dotProduct(e, crossResult) > 0.0)
			lambda = -lambda;
		corr1_x = -invMass1 * lambda * d0[0];
		corr1_y = -invMass1 * lambda * d0[1];
		corr1_z = -invMass1 * lambda * d0[2];

		//corr2 = -invMass2 * lambda * d1;
		corr2_x = -invMass2 * lambda * d1[0];
		corr2_y = -invMass2 * lambda * d1[1];
		corr2_z = -invMass2 * lambda * d1[2];

		//corr3 = -invMass3 * lambda * d2;
		corr3_x = -invMass3 * lambda * d2[0];
		corr3_y = -invMass3 * lambda * d2[1];
		corr3_z = -invMass3 * lambda * d2[2];
		//corr4 = -invMass4 * lambda * d3;
		corr4_x = -invMass4 * lambda * d3[0];
		corr4_y = -invMass4 * lambda * d3[1];
		corr4_z = -invMass4 * lambda * d3[2];

		return true;
	}

	__device__ bool Solve_D(int e, Edge g_edges[], Vertex vertices[]) {

		int i1 = g_edges[e].m_vert[0];
		int i2 = g_edges[e].m_vert[1];
		float rl = g_edges[e].rl;
		float st = g_edges[e].st;
		if (i1 == 0 && i2 == 0)return false;
		//printf("in Solve_D, e(edgeNumber) :%d vertexNumber1: %d, vertexNumber2: %d, rl : %f, st: %f\n",e, i1,i2, rl, st);
		float x1, y1, z1;
		float x2, y2, z2;
		x1 = vertices[i1].px;
		y1 = vertices[i1].py;
		z1 = vertices[i1].pz;

		//if(i1 == 8005) {
		//    printf("solveD: [%d] %f %f %f %f %f %f\n",8005, vertices[i1].px, vertices[i1].py, vertices[i1].pz, x1, y1, z1);
		//}

		x2 = vertices[i2].px;
		y2 = vertices[i2].py;
		z2 = vertices[i2].pz;

		float restLength = rl;
		float stiffness = st;
		const float invMass1 = vertices[i1].im;
		const float invMass2 = vertices[i2].im;

		float corr1_x, corr1_y, corr1_z;
		float corr2_x, corr2_y, corr2_z;

		//if(i1 == 9283){
		//    printf("Solve_D1: %f %f %f\n",x1, y1,z1);
		//    printf("Solve_D2: %f %f %f\n",vertices[9283].px, vertices[9283].py,vertices[9283].pz);
		//    
		//}
		//if(i2 == 9283){
		//    printf("Solve_D: %f %f %f",x2, y2,z2);
		//}
		bool res = Solve_DistanceConstraint(
			x1, y1, z1, invMass1,
			x2, y2, z2, invMass2,
			restLength,
			stiffness,
			corr1_x, corr1_y, corr1_z,
			corr2_x, corr2_y, corr2_z
			, i1, i2
		);
		if (res) {
			if (invMass1 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i1].dx += corr1_x;
				vertices[i1].dy += corr1_y;
				vertices[i1].dz += corr1_z;

				x1 += corr1_x;
				y1 += corr1_y;
				z1 += corr1_z;

				//printf("edge1[%d]: %f %f %f\n", e, corr1_x, corr1_y, corr1_z);
			}
			if (invMass2 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i2].dx += corr2_x;
				vertices[i2].dy += corr2_y;
				vertices[i2].dz += corr2_z;
				x2 += corr2_x;
				y2 += corr2_y;
				z2 += corr2_z;

				//printf("edge2[%d]: %f %f %f\n", e, corr2_x, corr2_y, corr2_z);
			}
		}
		vertices[i1].px = x1;
		vertices[i1].py = y1;
		vertices[i1].pz = z1;

		vertices[i2].px = x2;
		vertices[i2].py = y2;
		vertices[i2].pz = z2;
		return res;
	}

	__device__ bool Solve_SkinTensionConstraint(int e, SkinSurface g_skins[], Vertex vertices[]) {

		int i1 = g_skins[e].m_vert[0];
		int i2 = g_skins[e].m_vert[1];
		int i3 = g_skins[e].m_vert[2];
		float st = g_skins[e].st;
		float rsa = g_skins[e].rsa;

		float x1, y1, z1;
		float x2, y2, z2;
		float x3, y3, z3;

		x1 = vertices[i1].px;
		y1 = vertices[i1].py;
		z1 = vertices[i1].pz;

		x2 = vertices[i2].px;
		y2 = vertices[i2].py;
		z2 = vertices[i2].pz;

		x3 = vertices[i3].px;
		y3 = vertices[i3].py;
		z3 = vertices[i3].pz;


		float restSkinArea = rsa;
		float stiffness = st;
		const float invMass1 = vertices[i1].im;
		const float invMass2 = vertices[i2].im;
		const float invMass3 = vertices[i3].im;

		float corr1_x = 0, corr1_y = 0, corr1_z = 0;
		float corr2_x = 0, corr2_y = 0, corr2_z = 0;
		float corr3_x = 0, corr3_y = 0, corr3_z = 0;


		bool res = solve_SkinTensionConstraint(
			x1, y1, z1, invMass1,
			x2, y2, z2, invMass2,
			x3, y3, z3, invMass3,
			restSkinArea,
			stiffness,
			corr1_x, corr1_y, corr1_z,
			corr2_x, corr2_y, corr2_z,
			corr3_x, corr3_y, corr3_z
		);
		if (res) {
			if (invMass1 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i1].dx += corr1_x;
				vertices[i1].dy += corr1_y;
				vertices[i1].dz += corr1_z;

				x1 += corr1_x;
				y1 += corr1_y;
				z1 += corr1_z;
			}
			if (invMass2 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i2].dx += corr2_x;
				vertices[i2].dy += corr2_y;
				vertices[i2].dz += corr2_z;
				x2 += corr2_x;
				y2 += corr2_y;
				z2 += corr2_z;
			}
			if (invMass3 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i3].dx += corr3_x;
				vertices[i3].dy += corr3_y;
				vertices[i3].dz += corr3_z;
				x3 += corr3_x;
				y3 += corr3_y;
				z3 += corr3_z;
			}
		}
		vertices[i1].px = x1;
		vertices[i1].py = y1;
		vertices[i1].pz = z1;

		vertices[i2].px = x2;
		vertices[i2].py = y2;
		vertices[i2].pz = z2;

		vertices[i3].px = x3;
		vertices[i3].py = y3;
		vertices[i3].pz = z3;
		return res;
	}

	__device__ bool Solve_V(int e, Tet g_tets[], Vertex vertices[]) {

		int i1 = g_tets[e].m_vert[0];
		int i2 = g_tets[e].m_vert[1];
		int i3 = g_tets[e].m_vert[2];
		int i4 = g_tets[e].m_vert[3];
		float rv = g_tets[e].rv;
		float st = g_tets[e].st;
		if (i1 == 0 && i2 == 0)return false;
		//printf("in Solve_D, e(edgeNumber) :%d vertexNumber1: %d, vertexNumber2: %d, rl : %f, st: %f\n",e, i1,i2, rl, st);

		float x1, y1, z1;
		float x2, y2, z2;
		float x3, y3, z3;
		float x4, y4, z4;
		x1 = vertices[i1].px;
		y1 = vertices[i1].py;
		z1 = vertices[i1].pz;

		x2 = vertices[i2].px;
		y2 = vertices[i2].py;
		z2 = vertices[i2].pz;

		x3 = vertices[i3].px;
		y3 = vertices[i3].py;
		z3 = vertices[i3].pz;

		x4 = vertices[i4].px;
		y4 = vertices[i4].py;
		z4 = vertices[i4].pz;

		float restVolume = rv;
		float stiffness = st;

		const float invMass1 = vertices[i1].im;
		const float invMass2 = vertices[i2].im;
		const float invMass3 = vertices[i3].im;
		const float invMass4 = vertices[i4].im;

		float corr1_x, corr1_y, corr1_z;
		float corr2_x, corr2_y, corr2_z;
		float corr3_x, corr3_y, corr3_z;
		float corr4_x, corr4_y, corr4_z;

		//if(i1 == 9283){
		//    printf("Solve_D1: %f %f %f\n",x1, y1,z1);
		//    printf("Solve_D2: %f %f %f\n",vertices[9283].px, vertices[9283].py,vertices[9283].pz);
		//    
		//}
		//if(i2 == 9283){
		//    printf("Solve_D: %f %f %f",x2, y2,z2);
		//}
		bool res = solve_VolumeConstraint(
			x1, y1, z1, invMass1,
			x2, y2, z2, invMass2,
			x3, y3, z3, invMass3,
			x4, y4, z4, invMass4,
			restVolume,
			stiffness,
			corr1_x, corr1_y, corr1_z,
			corr2_x, corr2_y, corr2_z,
			corr3_x, corr3_y, corr3_z,
			corr4_x, corr4_y, corr4_z
		);

		if (res) {
			if (invMass1 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i1].dx += corr1_x;
				vertices[i1].dy += corr1_y;
				vertices[i1].dz += corr1_z;

				x1 += corr1_x;
				y1 += corr1_y;
				z1 += corr1_z;
			}
			if (invMass2 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i2].dx += corr2_x;
				vertices[i2].dy += corr2_y;
				vertices[i2].dz += corr2_z;
				x2 += corr2_x;
				y2 += corr2_y;
				z2 += corr2_z;
			}
			if (invMass3 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i3].dx += corr3_x;
				vertices[i3].dy += corr3_y;
				vertices[i3].dz += corr3_z;
				x3 += corr3_x;
				y3 += corr3_y;
				z3 += corr3_z;
			}
			if (invMass4 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i4].dx += corr4_x;
				vertices[i4].dy += corr4_y;
				vertices[i4].dz += corr4_z;
				x4 += corr4_x;
				y4 += corr4_y;
				z4 += corr4_z;
			}
		}
		vertices[i1].px = x1;
		vertices[i1].py = y1;
		vertices[i1].pz = z1;

		vertices[i2].px = x2;
		vertices[i2].py = y2;
		vertices[i2].pz = z2;

		vertices[i3].px = x3;
		vertices[i3].py = y3;
		vertices[i3].pz = z3;

		vertices[i4].px = x4;
		vertices[i4].py = y4;
		vertices[i4].pz = z4;

		return res;
	}

	__device__ bool Solve_Dihed(int e, Tet g_tets[], Vertex vertices[]) {
		int i1 = g_tets[e].m_vert[0];
		int i2 = g_tets[e].m_vert[1];
		int i3 = g_tets[e].m_vert[2];
		int i4 = g_tets[e].m_vert[3];
		float ra = g_tets[e].ra;
		float st = g_tets[e].st;
		if (i1 == 0 && i2 == 0) return false;
		//printf("in Solve_D, e(edgeNumber) :%d vertexNumber1: %d, vertexNumber2: %d, rl : %f, st: %f\n",e, i1,i2, rl, st);

		float x1, y1, z1;
		float x2, y2, z2;
		float x3, y3, z3;
		float x4, y4, z4;
		x1 = vertices[i1].px;
		y1 = vertices[i1].py;
		z1 = vertices[i1].pz;

		x2 = vertices[i2].px;
		y2 = vertices[i2].py;
		z2 = vertices[i2].pz;

		x3 = vertices[i3].px;
		y3 = vertices[i3].py;
		z3 = vertices[i3].pz;

		x4 = vertices[i4].px;
		y4 = vertices[i4].py;
		z4 = vertices[i4].pz;

		float restAngle = ra;
		float stiffness = st;

		const float invMass1 = vertices[i1].im;
		const float invMass2 = vertices[i2].im;
		const float invMass3 = vertices[i3].im;
		const float invMass4 = vertices[i4].im;

		float corr1_x, corr1_y, corr1_z;
		float corr2_x, corr2_y, corr2_z;
		float corr3_x, corr3_y, corr3_z;
		float corr4_x, corr4_y, corr4_z;

		//if(i1 == 9283){
		//    printf("Solve_D1: %f %f %f\n",x1, y1,z1);
		//    printf("Solve_D2: %f %f %f\n",vertices[9283].px, vertices[9283].py,vertices[9283].pz);
		//    
		//}
		//if(i2 == 9283){
		//    printf("Solve_D: %f %f %f",x2, y2,z2);
		//}
		bool res = solve_DihedralConstraint(
			x1, y1, z1, invMass1,
			x2, y2, z2, invMass2,
			x3, y3, z3, invMass3,
			x4, y4, z4, invMass4,
			restAngle,
			stiffness,
			corr1_x, corr1_y, corr1_z,
			corr2_x, corr2_y, corr2_z,
			corr3_x, corr3_y, corr3_z,
			corr4_x, corr4_y, corr4_z
		);

		if (res) {
			if (invMass1 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i1].dx += corr1_x;
				vertices[i1].dy += corr1_y;
				vertices[i1].dz += corr1_z;

				x1 += corr1_x;
				y1 += corr1_y;
				z1 += corr1_z;
			}
			if (invMass2 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i2].dx += corr2_x;
				vertices[i2].dy += corr2_y;
				vertices[i2].dz += corr2_z;
				x2 += corr2_x;
				y2 += corr2_y;
				z2 += corr2_z;
			}
			if (invMass3 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i3].dx += corr3_x;
				vertices[i3].dy += corr3_y;
				vertices[i3].dz += corr3_z;
				x3 += corr3_x;
				y3 += corr3_y;
				z3 += corr3_z;
			}
			if (invMass4 != 0.0) {
				// DONE: 각 정점의 dx, dy, dz를 누적
				vertices[i4].dx += corr4_x;
				vertices[i4].dy += corr4_y;
				vertices[i4].dz += corr4_z;
				x4 += corr4_x;
				y4 += corr4_y;
				z4 += corr4_z;
			}
		}
		vertices[i1].px = x1;
		vertices[i1].py = y1;
		vertices[i1].pz = z1;

		vertices[i2].px = x2;
		vertices[i2].py = y2;
		vertices[i2].pz = z2;

		vertices[i3].px = x3;
		vertices[i3].py = y3;
		vertices[i3].pz = z3;

		vertices[i4].px = x4;
		vertices[i4].py = y4;
		vertices[i4].pz = z4;

		return res;

	}

	__global__ void solveKernel_Distance(int CG_SIZE, int d_group[], Edge d_edges[], Vertex d_vertices[]) {
		//printf("i'm kernel\n");
		int i = blockIdx.x * 256 + threadIdx.x; // 17, 723 -> 17723
		//printf("i = %d\n", i );
//printf("gpu kernel is called!!!\n");
		if (i >= CG_SIZE)
			return;
		int e_index = d_group[i];




		Solve_D(e_index, d_edges, d_vertices);

		//__syncthreads();
	}
	//thth
	__global__ void solveKernel_SkinTension(int SG_SIZE, int d_group[], SkinSurface d_skins[], Vertex d_vertices[]) {
		//printf("i'm kernel\n");
		int i = blockIdx.x * 256 + threadIdx.x; // 17, 723 -> 17723
		//printf("i = %d\n", i );
//printf("gpu kernel is called!!!\n");
		if (i >= SG_SIZE)
			return;
		int e_index = d_group[i];




		Solve_SkinTensionConstraint(e_index, d_skins, d_vertices);

		//__syncthreads();
	}

	__global__ void solveKernel_Volume(int CG_SIZE, int d_group[], Tet d_tets[], Vertex d_vertices[]) {
		int i = blockIdx.x * 256 + threadIdx.x;
		if (i >= CG_SIZE) {
			//printf("여기?ㅋㅋ");
			return;
		}
		int t_index = d_group[i];

		//printf("여긴 옴?");

		Solve_V(t_index, d_tets, d_vertices);

	}

	__global__ void solveKernel_Dihedral(int CG_SIZE, int d_group[], Tet d_tets[], Vertex d_vertices[]) {
		int i = blockIdx.x * 256 + threadIdx.x;
		if (i >= CG_SIZE)
			return;
		int t_index = d_group[i];

		Solve_Dihed(t_index, d_tets, d_vertices);
	}

	int Testbed::cuda_SolveDistanceConstraint(int Dgroup[], int CG_SIZE) {
		//printf("hi cuda\n");

		cudaError_t cudaStatus;

		cudaStatus = cudaMemcpy(Testbed::d_DistanceGroup, Dgroup, CG_SIZE * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMemcpy failed3!\n");
			return cudaStatus;
		}
		int b = 1, t = CG_SIZE; // CG_SIZE 38,713
		if (CG_SIZE > 256) {
			b = (CG_SIZE + 255) / 256; // 39
			t = 256; // 1000
		}

		//printf("[%d] group size=%d <<<%d,%d>>> \n ", group_id, CG_SIZE, b, t);
		solveKernel_Distance << <b, t >> > (CG_SIZE, Testbed::d_DistanceGroup, d_edges, d_vertices);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "DsolveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}



		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) {
		//    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		//    return cudaStatus;
		//}
		//free(h_group);

		return 0;
	}

	int Testbed::cuda_SolveVolumeConstraint(int Vgroup[], int T_CG_SIZE) {
		//printf("hi cuda\n");

		cudaError_t cudaStatus;


		int b = 1, t = T_CG_SIZE; // CG_SIZE 38,713
		if (T_CG_SIZE > 256) {
			b = (T_CG_SIZE + 255) / 256; // 39
			t = 256; // 1000
		}
		//@소진
		//if (T_CG_SIZE > 512) { //커널 블록 수 안줄이면 터집니다... 근데 줄이면 프로그램이 터짐..니다.. - thth
		//	b = (T_CG_SIZE + 511) / 512; // 39
		//	t = 512; // 1000
		//}

		//printf("[%d] group size=%d <<<%d,%d>>> \n ", group_id, CG_SIZE, b, t);

		cudaStatus = cudaMemcpy(Testbed::d_VolumeGroup, Vgroup, T_CG_SIZE * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMemcpy failed4!\n");
			return cudaStatus;
		}

		solveKernel_Volume << <b, t >> > (T_CG_SIZE, Testbed::d_VolumeGroup, d_tets, d_vertices);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "VsolveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) {
		//    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		//    return cudaStatus;
		//}
		//free(h_group);

		return 0;
	}

	int Testbed::cuda_SolveDihedralConstraint(int Vgroup[], int T_CG_SIZE) {
		//printf("hi cuda\n");

		cudaError_t cudaStatus;


		int b = 1, t = T_CG_SIZE; // CG_SIZE 38,713
		if (T_CG_SIZE > 128) {
			b = (T_CG_SIZE + 127) / 128; // 39
			t = 128; // 1000
		}

		//printf("[%d] group size=%d <<<%d,%d>>> \n ", group_id, CG_SIZE, b, t);

		cudaStatus = cudaMemcpy(Testbed::d_VolumeGroup, Vgroup, T_CG_SIZE * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMemcpy failed5!\n");
			return cudaStatus;
		}

		solveKernel_Dihedral << <b, t >> > (T_CG_SIZE, Testbed::d_VolumeGroup, d_tets, d_vertices);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "VsolveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) {
		//    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		//    return cudaStatus;
		//}
		//free(h_group);

		return 0;
	}

	int Testbed::cuda_SolveSkinTensionConstraint(int Sgroup[], int SG_SIZE) { //CG_SIZE에 특별한 의미?
		cudaError_t cudaStatus;

		cudaStatus = cudaMemcpy(Testbed::d_SkinGroup, Sgroup, SG_SIZE * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("group cudaMemcpy failed6!\n");
			return cudaStatus;
		}
		int b = 1, t = SG_SIZE; // CG_SIZE 38,713
		if (SG_SIZE > 256) {
			b = (SG_SIZE + 255) / 256; // 39
			t = 256; // 1000
		}

		//printf("[%d] group size=%d <<<%d,%d>>> \n ", group_id, CG_SIZE, b, t);
		solveKernel_SkinTension << <b, t >> > (SG_SIZE, Testbed::d_SkinGroup, d_skins, d_vertices);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "SkinsolveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}



		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) {
		//    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		//    return cudaStatus;
		//}
		//free(h_group);

		return 0;
	}

	class vecsm {
	public:
		__host__ __device__ vecsm() {}
		__host__ __device__ vecsm(float e0, float e1, float e2) {
			m[0] = e0;
			m[1] = e1;
			m[2] = e2;
		}

		__host__ __device__ inline float x() const { return m[0]; }
		__host__ __device__ inline float y() const { return m[1]; }
		__host__ __device__ inline float z() const { return m[2]; }

		__host__ __device__ inline float r() const { return m[0]; }
		__host__ __device__ inline float g() const { return m[1]; }
		__host__ __device__ inline float b() const { return m[2]; }

		__host__ __device__ inline const vecsm& operator+() const { return *this; }
		__host__ __device__ inline vecsm operator-() const { return vecsm(-m[0], -m[1], -m[2]); }
		__host__ __device__ inline float operator[] (int i) const { return m[i]; }
		__host__ __device__ inline float& operator[] (int i) { return m[i]; }

		__host__ __device__ inline float4 GetFloat4(float w = 1) {
			float4 a = { m[0], m[1], m[2], w };
			//a.x = m[0];
			//a.y = m[1];
			//a.z = m[2];
			//a.w = w;
			return a;
		}

		__host__ __device__ inline vecsm& operator+=(const vecsm& v) {
			m[0] += v.m[0];
			m[1] += v.m[1];
			m[2] += v.m[2];
			return *this;
		}

		__host__ __device__ inline vecsm& operator-=(const vecsm& v) {
			m[0] -= v.m[0];
			m[1] -= v.m[1];
			m[2] -= v.m[2];
			return *this;
		}

		__host__ __device__ inline vecsm& operator*=(const vecsm& v) {
			m[0] *= v.m[0];
			m[1] *= v.m[1];
			m[2] *= v.m[2];
			return *this;
		}

		__host__ __device__ inline vecsm& operator/=(const vecsm& v) {
			m[0] /= v.m[0];
			m[1] /= v.m[1];
			m[2] /= v.m[2];
			return *this;
		}

		__host__ __device__ inline vecsm& operator*=(const float t) {
			m[0] *= t;
			m[1] *= t;
			m[2] *= t;
			return *this;
		}

		__host__ __device__ inline vecsm& operator/=(const float t) {
			float k = 1.0 / t;
			m[0] *= k;
			m[1] *= k;
			m[2] *= k;
			return *this;
		}

		__host__ __device__ inline float length() const {
			return sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
		}

		__host__ __device__ inline float squared_length() const {
			return (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
		}

		__host__ __device__ inline void make_unit_vector() {
			float k = 1.0 / (sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]));
			m[0] *= k;
			m[1] *= k;
			m[2] *= k;
		};

		__host__ __device__ inline vecsm CrossProduct(vecsm b) {
			return vecsm(
				m[1] * b.m[2] - m[2] * b.m[1],
				m[2] * b.m[0] - m[0] * b.m[2],
				m[0] * b.m[1] - m[1] * b.m[0]);
		}
		void Print() {
			printf("(%f, %f, %f)\n", m[0], m[1], m[2]);
		}
		float m[3];

	};

	__host__ __device__ inline vecsm operator+(const vecsm& v1, const vecsm& v2) {
		return vecsm(v1.m[0] + v2.m[0], v1.m[1] + v2.m[1], v1.m[2] + v2.m[2]);
	}

	__host__ __device__ inline vecsm operator-(const vecsm& v1, const vecsm& v2) {
		return vecsm(v1.m[0] - v2.m[0], v1.m[1] - v2.m[1], v1.m[2] - v2.m[2]);
	}

	__host__ __device__ inline vecsm operator*(const vecsm& v1, const vecsm& v2) {
		return vecsm(v1.m[0] * v2.m[0], v1.m[1] * v2.m[1], v1.m[2] * v2.m[2]);
	}

	__host__ __device__ inline vecsm operator/(const vecsm& v1, const vecsm& v2) {
		return vecsm(v1.m[0] / v2.m[0], v1.m[1] / v2.m[1], v1.m[2] / v2.m[2]);
	}

	__host__ __device__ inline vecsm operator*(const float t, const vecsm& v) {
		return vecsm(t * v.m[0], t * v.m[1], t * v.m[2]);
	}

	__host__ __device__ inline vecsm operator/(const vecsm& v, const float t) {
		return vecsm(v.m[0] / t, v.m[1] / t, v.m[2] / t);
	}

	__host__ __device__ inline vecsm operator*(const vecsm& v, const float t) {
		return vecsm(v.m[0] * t, v.m[1] * t, v.m[2] * t);
	}

	__host__ __device__ inline float dot(const vecsm& v1, const vecsm& v2) {
		return (v1.m[0] * v2.m[0] + v1.m[1] * v2.m[1] + v1.m[2] * v2.m[2]);
	}

	__host__ __device__ inline vecsm cross(const vecsm& v1, const vecsm& v2) {
		return vecsm(
			(v1.m[1] * v2.m[2] - v1.m[2] * v2.m[1]),
			-(v1.m[0] * v2.m[2] - v1.m[2] * v2.m[0]),
			(v1.m[0] * v2.m[1] - v1.m[1] * v2.m[0])
		);
	}

	__host__ __device__ inline vecsm unitVector(vecsm v) {
		return (v / v.length());
	}

	__constant__ vec3 deye[1];
	__constant__ vec3 duvw[3];
	//vecsm eye(0, 200, 256);
	//vecsm at(128, 128, 128);
	//vecsm dirsm(0,0,0);
	//vecsm up(0, -1, 0);
	//vecsm w;
	//vecsm u;
	//vecsm v;
	//vecsm rayStart;


	//void MakeAlphaTable(int a, int b) {
	//	for (int i = 0; i < 256; i++) {
	//		if (i < a) {
	//			AlphaTable[i] = 0;
	//		}
	//		else if (i >= a && i < b) {
	//			AlphaTable[i] = (float)(i - a) / (b - a);
	//		}
	//		else if (i >= b) {
	//			AlphaTable[i] = 1;
	//		}
	//	}
	//}
	// void samplingRGB(float x, float y, float z, float* color) { // 일종의 colorTable 역할
	//	if (x >= 255 || y >= 255 || z >= 255 || x < 0 || y < 0 || z < 0) {
	//		color[0] = color[1] = color[2] = 0;
	//		return;
	//	}
	//	int ix = (int)x;
	//	float wx = x - ix;
	//	int iy = (int)y;
	//	float wy = y - iy;
	//	int iz = (int)z;
	//	float wz = z - iz;

	//	for (int i = 0; i < 3; i++) { // 각 색상 채널에 대해 별도로 보간
	//		float a = (i == 0 ? volume[iz][iy][ix].r : (i == 1 ? volume[iz][iy][ix].g : volume[iz][iy][ix].b));
	//		float b = (i == 0 ? volume[iz][iy][ix + 1].r : (i == 1 ? volume[iz][iy][ix + 1].g : volume[iz][iy][ix + 1].b));
	//		float c = (i == 0 ? volume[iz][iy + 1][ix].r : (i == 1 ? volume[iz][iy + 1][ix].g : volume[iz][iy + 1][ix].b));
	//		float d = (i == 0 ? volume[iz][iy + 1][ix + 1].r : (i == 1 ? volume[iz][iy + 1][ix + 1].g : volume[iz][iy + 1][ix + 1].b));
	//		float e = (i == 0 ? volume[iz + 1][iy][ix].r : (i == 1 ? volume[iz + 1][iy][ix].g : volume[iz + 1][iy][ix].b));
	//		float f = (i == 0 ? volume[iz + 1][iy][ix + 1].r : (i == 1 ? volume[iz + 1][iy][ix + 1].g : volume[iz + 1][iy][ix + 1].b));
	//		float g = (i == 0 ? volume[iz + 1][iy + 1][ix].r : (i == 1 ? volume[iz + 1][iy + 1][ix].g : volume[iz + 1][iy + 1][ix].b));
	//		float h = (i == 0 ? volume[iz + 1][iy + 1][ix + 1].r : (i == 1 ? volume[iz + 1][iy + 1][ix + 1].g : volume[iz + 1][iy + 1][ix + 1].b));

	//		color[i] = a * (1 - wx) * (1 - wy) * (1 - wz) + b * (wx) * (1 - wy) * (1 - wz) +
	//			c * (1 - wx) * wy * (1 - wz) + d * wx * wy * (1 - wz) + e * (1 - wx) * (1 - wy) * wz
	//			+ f * (wx) * (1 - wy) * wz + g * (1 - wx) * wy * wz + h * wx * wy * wz;
	//	}
	//}

	// float sampling(float x, float y, float z) { // x=3.7 , ix = 3
	//	// 힌트, 좌표 벗어나서 메모리 참조 오류 해결할 것.
	//	if (x >= 255 || y >= 255 || z >= 255 || x < 0 || y < 0 || z < 0)
	//		return 0;
	//	int ix = (int)x;
	//	float wx = x - ix;
	//	int iy = (int)y;
	//	float wy = y - iy;
	//	int iz = (int)z;
	//	float wz = z - iz;


	//	float a = volume[iz][iy][ix].a;
	//	float b = volume[iz][iy][ix + 1].a;
	//	float c = volume[iz][iy + 1][ix].a;
	//	float d = volume[iz][iy + 1][ix + 1].a;
	//	float e = volume[iz + 1][iy][ix].a;
	//	float f = volume[iz + 1][iy][ix + 1].a;
	//	float g = volume[iz + 1][iy + 1][ix].a;
	//	float h = volume[iz + 1][iy + 1][ix + 1].a;

	//	float res = a * (1 - wx) * (1 - wy) * (1 - wz) + b * (wx) * (1 - wy) * (1 - wz) +
	//		c * (1 - wx) * wy * (1 - wz) + d * wx * wy * (1 - wz) + e * (1 - wx) * (1 - wy) * wz
	//		+ f * (wx) * (1 - wy) * wz + g * (1 - wx) * wy * wz + h * wx * wy * wz;


	//	if (res > 255) res = 255;
	//	//반환 값에 따라 선명도 조절
	//	return res; // 77.285

	//	//return vol[iz][iy][ix];

	//	//float a = vol[iz][iy][ix], float b=[iz][iy][ix+1], float c, float d
	//	//float res = a* (1 - wx)* (1 - wy) + b*(wx) * (1 - wy) +
	//	//   c * (1 - wx) * wy + d * wx * wy;
	//	//return res; // 77.285
	//}
#include <windows.h>
#include <iostream>

	// void FillMyTexture() {
	//	// Fill MyTexture with grayscale gradient
	//	//FILE* fp = fopen("C:/Users/admin/instant-ngp/src/sm0305(2).den", "rb");
	//	////fread(volume, sizeof(voxel1), 256 * 256 * 256, fp);
	//	//fread(volume, sizeof(float4), 256 * 256 * 256, fp);
	//	//fclose(fp);

	//	for (int iy = 0; iy < HEIGHT; iy++) {
	//		for (int ix = 0; ix < WIDTH; ix++) {
	//			int dx = (ix - 512) / 2; // 중심점 기준으로 영상 u축 방향으로 몇칸 움직였나
	//			int dy = (iy - 512) / 2; // 중심점 기준으로 영상 v축 방향으로 몇칸 움직였나

	//			//vec3 rs = eye + u * dx + v * dy;
	//			rayStart = eye + u * dx + v * dy;
	//			int maxv = 0;
	//			float alphasum = 0;
	//			float colorsum[3];
	//			for (int t = 0; t < 3; t++) {
	//				colorsum[t] = 0;
	//			}
	//			for (int t = 0; t < 255; t++) {
	//				// vec3 s = rs + w*t;
	//				vecsm s;
	//				s.m[0] = rayStart.m[0] + w.m[0] * t;
	//				s.m[1] = rayStart.m[1] + w.m[1] * t;
	//				s.m[2] = rayStart.m[2] + w.m[2] * t;
	//				//float vz = t, vy = iy, vx = ix; // ity % 256, vy = ix % 256, vx = t;
	//				//if (vz >= 225)
	//				//   continue;
	//				GLubyte Intensity = sampling(s.m[0], s.m[1], s.m[2]); // samplig(s);
	//				//float max = __max(max, Intensity);

	//				float color[3];
	//				samplingRGB(s.m[0], s.m[1], s.m[2], color);

	//				float alpha = AlphaTable[Intensity];
	//				if (alpha == 0) {
	//					continue;
	//				}
	//				float dx = (sampling(s.m[0] + 1, s.m[1], s.m[2]) - sampling(s.m[0] - 1, s.m[1], s.m[2])) * 0.5;
	//				float dy = (sampling(s.m[0], s.m[1] + 1, s.m[2]) - sampling(s.m[0], s.m[1] - 1, s.m[2])) * 0.5;
	//				float dz = (sampling(s.m[0], s.m[1], s.m[2] + 1) - sampling(s.m[0], s.m[1], s.m[2] - 1)) * 0.5;
	//				vecsm N;
	//				N.m[0] = dx;
	//				N.m[1] = dy;
	//				N.m[2] = dz;
	//				N.vec_norm();
	//				// alpha blending
	//				// calc light
	//				float Ia = 0.3, Id = 0.6, Is = 0.3; // 고정, Ia[3] = {0.3 , 0.3, 0.3}; // r,g,b동일 느낌
	//				float Ka[3], Kd[3];
	//				for (int i = 0; i < 3; i++) {
	//					Ka[i] = color[i];
	//					Kd[i] = color[i];
	//				}
	//				float Ks = 1.0;
	//				vecsm L(w); // 광부 조명 모델(편리)
	//				L = L * -1.0;

	//				float NL = N.dot(L);
	//				if (NL < 0)
	//					NL = -NL;
	//				vecsm H = eye; // 광부모델의 경우만 가능 원래는 (L + eye).normalize();
	//				float NH = N.dot(H); // RV
	//				if (NH < 0)
	//					NH = -NH;
	//				vecsm R = N * (2 * NL) - L;
	//				R.vec_norm();
	//				float RV = fabs(R.dot(w));
	//				const float po = 10;
	//				// K는 물체색상, 그러므로 color[3]을 사용.
	//				float I[3];
	//				float e[3];
	//				for (int i = 0; i < 3; i++) {

	//					I[i] = Ia * Ka[i] + Id * Kd[i] * NL + Is * Ks * pow(RV, po);
	//					e[i] = I[i] * alpha;
	//					colorsum[i] = colorsum[i] + e[i] * (1 - alphasum);
	//				}


	//				alphasum = alphasum + alpha * (1 - alphasum);
	//				// 속도향상: 화질을 저하없이 (최소한으로하면서) 연산을 줄이는 방법 연구
	//				if (alphasum > 0.99) {
	//					break;
	//				}

	//			}

	//			for (int i = 0; i < 3; i++) {
	//				if (colorsum[i] > 1)
	//					colorsum[i] = 1;
	//			}

	//			MyTexture[iy][ix][0] = colorsum[0] * 255.0; //Red 값에 할당
	//			MyTexture[iy][ix][1] = colorsum[1] * 255.0; //Green 값에 할당
	//			MyTexture[iy][ix][2] = colorsum[2] * 255.0; //Blue 값에 할당

	//		}
	//	}
	//}
	void generateGradient() {

		//glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glGenTextures(1, &texID);
		glBindTexture(GL_TEXTURE_2D, texID);
		//FillMyTexture();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Window_WIDTH, Window_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // GL_REPEAT
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // GL_REPEAT
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glClear(GL_COLOR_BUFFER_BIT);

		// Define vertex data
		float vertices[] = {
			-1.0f, -1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f
		};

		// Define texture coordinates
		float texCoords[] = {
			0.0f, 1.0f,
			0.0f, 0.0f,
			1.0f, 0.0f,
			1.0f, 1.0f
		};

		// Generate vertex buffer object (VBO) for vertices
		GLuint vboVertices;
		glGenBuffers(1, &vboVertices);
		glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		// Generate vertex buffer object (VBO) for texture coordinates
		GLuint vboTexCoords;
		glGenBuffers(1, &vboTexCoords);
		glBindBuffer(GL_ARRAY_BUFFER, vboTexCoords);
		glBufferData(GL_ARRAY_BUFFER, sizeof(texCoords), texCoords, GL_STATIC_DRAW);

		// Define vertex and fragment shaders
		const char* vertexShaderSource = "#version 330 core\n"
			"layout (location = 0) in vec3 aPos;\n"
			"layout (location = 1) in vec2 aTexCoord;\n"
			"out vec2 TexCoord;\n"
			"void main() {\n"
			"    gl_Position = vec4(aPos, 1.0);\n"
			"    TexCoord = aTexCoord;\n"
			"}\0";
		const char* fragmentShaderSource = "#version 330 core\n"
			"out vec4 FragColor;\n"
			"in vec2 TexCoord;\n"
			"uniform sampler2D textureSampler;\n"
			"void main() {\n"
			"    FragColor = texture(textureSampler, TexCoord);\n"
			"}\0";

		// Compile vertex shader
		GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
		glCompileShader(vertexShader);

		// Compile fragment shader
		GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
		glCompileShader(fragmentShader);

		// Create shader program
		GLuint shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);
		glUseProgram(shaderProgram);

		// Bind vertex attributes
		glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, vboTexCoords);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);

		// Set texture uniform
		glUniform1i(glGetUniformLocation(shaderProgram, "textureSampler"), 0);

		// Generate texture


		// Render using vertex and fragment shaders
		glDrawArrays(GL_QUADS, 0, 4);

		// Cleanup
		glDeleteBuffers(1, &vboVertices);
		glDeleteBuffers(1, &vboTexCoords);
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
		glDeleteProgram(shaderProgram);

		// Swap buffers
		//glfwSwapBuffers(window);
	}
	__device__ float4 TLSampling(vecsm sample, float4* vol) {
		int index = int(sample.z()) * 256 * 256 + int(sample.y()) * 256 + int(sample.x());
		int ix = index % 256;
		int iy = (index / 256) % 256;
		int iz = index / (256 * 256);

		float wx = sample.x() - ix;
		float wy = sample.y() - iy;
		float wz = sample.z() - iz;

		float4 rgba;

		if (ix >= 255 || iy >= 255 || iz >= 255 || ix < 0 || iy < 0 || iz < 0)
			return rgba;

		float4 a = vol[iz * 256 * 256 + iy * 256 + ix];
		float4 b = vol[iz * 256 * 256 + iy * 256 + ix + 1];
		float4 c = vol[iz * 256 * 256 + (iy + 1) * 256 + ix];
		float4 d = vol[iz * 256 * 256 + (iy + 1) * 256 + ix + 1];
		float4 e = vol[(iz + 1) * 256 * 256 + iy * 256 + ix];
		float4 f = vol[(iz + 1) * 256 * 256 + iy * 256 + ix + 1];
		float4 g = vol[(iz + 1) * 256 * 256 + (iy + 1) * 256 + ix];
		float4 h = vol[(iz + 1) * 256 * 256 + (iy + 1) * 256 + ix + 1];

		rgba.x = a.x * (1 - wx) * (1 - wy) * (1 - wz) + b.x * wx * (1 - wy) * (1 - wz) +
			c.x * (1 - wx) * wy * (1 - wz) + d.x * wx * wy * (1 - wz) +
			e.x * (1 - wx) * (1 - wy) * wz + f.x * wx * (1 - wy) * wz +
			g.x * (1 - wx) * wy * wz + h.x * wx * wy * wz;

		rgba.y = a.y * (1 - wx) * (1 - wy) * (1 - wz) + b.y * wx * (1 - wy) * (1 - wz) +
			c.y * (1 - wx) * wy * (1 - wz) + d.y * wx * wy * (1 - wz) +
			e.y * (1 - wx) * (1 - wy) * wz + f.y * wx * (1 - wy) * wz +
			g.y * (1 - wx) * wy * wz + h.y * wx * wy * wz;

		rgba.z = a.z * (1 - wx) * (1 - wy) * (1 - wz) + b.z * wx * (1 - wy) * (1 - wz) +
			c.z * (1 - wx) * wy * (1 - wz) + d.z * wx * wy * (1 - wz) +
			e.z * (1 - wx) * (1 - wy) * wz + f.z * wx * (1 - wy) * wz +
			g.z * (1 - wx) * wy * wz + h.z * wx * wy * wz;

		rgba.w = a.w * (1 - wx) * (1 - wy) * (1 - wz) + b.w * wx * (1 - wy) * (1 - wz) +
			c.w * (1 - wx) * wy * (1 - wz) + d.w * wx * wy * (1 - wz) +
			e.w * (1 - wx) * (1 - wy) * wz + f.w * wx * (1 - wy) * wz +
			g.w * (1 - wx) * wy * wz + h.w * wx * wy * wz;

		return rgba;
	}
	__device__ void RBInter(vec3 start, float trange[2]) {
		float txm = -1000000, txM = 100000;
		if (duvw[2].x != 0) {
			float tx1, tx2;
			tx1 = (VOLX - 1 - start.x) / duvw[2].x;
			tx2 = (0.5 - start.x) / duvw[2].x;
			txm = __min(tx1, tx2);
			txM = __max(tx1, tx2);
		}
		else {
			if (start.x < 0 || start.x > VOLX - 1) return;

		}

		float tym = -1000000, tyM = 100000;
		if (duvw[2].y != 0) {
			float ty1, ty2;
			ty1 = (VOLY - 1 - start.y) / duvw[2].y;
			ty2 = (0.5 - start.y) / duvw[2].y;

			tym = __min(ty1, ty2);
			tyM = __max(ty1, ty2);
		}

		else {
			if (start.z < 0 || start.z > VOLY - 1) return;

		}

		float tzm = -1000000, tzM = 100000;
		if (duvw[2].z != 0) {
			float tz1 = (VOLZ - 1 - start.z) / duvw[2].z;
			float tz2 = (0.5 - start.z) / duvw[2].z;
			tzm = __min(tz1, tz2);
			tzM = __max(tz1, tz2);
		}
		else {
			if (start.z < 0 || start.z > VOLZ - 1) return;

		}
		trange[0] = __max(__max(txm, tym), tzm);
		trange[1] = __min(__min(txM, tyM), tzM);

	}
	__device__ void GetAABB(const vec3& v1, const vec3& v2, const vec3& v3, const vec3& v4, vec3& Min, vec3& Max) {

		Min.x = fminf(fminf(v1.x, v2.x), fminf(v3.x, v4.x));
		Max.x = fmaxf(fmaxf(v1.x, v2.x), fmaxf(v3.x, v4.x));

		// Y 축에 대한 최소 및 최대값 계산
		Min.y = fminf(fminf(v1.y, v2.y), fminf(v3.y, v4.y));
		Max.y = fmaxf(fmaxf(v1.y, v2.y), fmaxf(v3.y, v4.y));

		// Z 축에 대한 최소 및 최대값 계산
		Min.z = fminf(fminf(v1.z, v2.z), fminf(v3.z, v4.z));
		Max.z = fmaxf(fmaxf(v1.z, v2.z), fmaxf(v3.z, v4.z));

	}
	__device__ bool OutsideTest(float4 b) {
		if (b.x < 0 || b.x>1 || b.y < 0 || b.y>1 || b.z < 0 || b.z>1 || b.w < 0 || b.w > 1)
			return true;
		else
			return false;
	}

	__device__ bool OutsideVol(int x, int y, int z) {
		if (x >= VOLX || y >= VOLY || z >= VOLZ || x < 0 || y < 0 || z < 0)
			return true;
		else
			return false;

	}

	__device__ inline void SampleTetrahedron(//mat4_sm X0,
		const vec3 A[2], const vec3 B[2], const vec3 C[2], const vec3 D[2],
		float4* pOutGrid, cudaTextureObject_t voltexObj, int forDebug = 0) {
		//printf("B: %2f, %2f, %2f, %2f, %2f, %2f\n", B[0].x, B[0].y, B[0].z, B[1].x, B[1].y, B[1].z);
		vec3 Min;
		vec3 Max;
		GetAABB(A[1], B[1], C[1], D[1], Min, Max);

		mat4 X1(
			A[1].x - D[1].x, B[1].x - D[1].x, C[1].x - D[1].x, 0.0f,
			A[1].y - D[1].y, B[1].y - D[1].y, C[1].y - D[1].y, 0.0f,
			A[1].z - D[1].z, B[1].z - D[1].z, C[1].z - D[1].z, 0.0f,
			1.0f, 1.0f, 1.0f, 1.0f);
		//A[1][0] - D[1][0], B[1][0] - D[1][0], C[1][0] - D[1][0], 0.0f,
		//A[1][1] - D[1][1], B[1][1] - D[1][1], C[1][1] - D[1][1], 0.0f,
		//A[1][2] - D[1][2], B[1][2] - D[1][2], C[1][2] - D[1][2], 0.0f
		mat4 X1Inv = inverse(X1);
		//auto det = X1[0][0] * (X1[1][1] * X1[2][2] - X1[1][2] * X1[2][1]) - X1[1][0] * (X1[0][1] * X1[2][2] - X1[2][1] * X1[0][2]) + X1[2][0] * (X1[0][1] * X1[1][2] - X1[1][1] * X1[0][2]);
		//auto invdet = 1 / det;
		//mat4 X1Inv;
		//X1Inv[0][0] = (X1[1][1] * X1[2][2] - X1[1][2] * X1[2][1]) * invdet;
		//X1Inv[1][0] = (X1[2][0] * X1[1][2] - X1[1][0] * X1[2][2]) * invdet;
		//X1Inv[2][0] = (X1[1][0] * X1[2][1] - X1[2][0] * X1[1][1]) * invdet;
		//X1Inv[0][1] = (X1[2][1] * X1[0][2] - X1[0][1] * X1[2][2]) * invdet;
		//X1Inv[1][1] = (X1[0][0] * X1[2][2] - X1[2][0] * X1[0][2]) * invdet;
		//X1Inv[2][1] = (X1[0][1] * X1[2][0] - X1[0][0] * X1[2][1]) * invdet;
		//X1Inv[0][2] = (X1[0][1] * X1[1][2] - X1[0][2] * X1[1][1]) * invdet;
		//X1Inv[1][2] = (X1[0][2] * X1[1][0] - X1[0][0] * X1[1][2]) * invdet;
		//X1Inv[2][2] = (X1[0][0] * X1[1][1] - X1[0][1] * X1[1][0]) * invdet;

		//mat4_sm X1Inv = X1.inverse();
		int3 iMin, iMax;
		iMin.x = (int)ceilf(Min.x);
		iMax.x = (int)floorf(Max.x);
		iMin.y = (int)ceilf(Min.y);
		iMax.y = (int)floorf(Max.y);
		iMin.z = (int)ceilf(Min.z);
		iMax.z = (int)floorf(Max.z);

		for (int z = iMin.z; z <= iMax.z; z++) {
			for (int y = iMin.y; y <= iMax.y; y++) {
				for (int x = iMin.x; x <= iMax.x; x++) {

					if (OutsideVol(x, y, z))
						continue;
					vec3 x1{ x - D[1].x, y - D[1].y, z - D[1].z };
					float4 x1_float4{ x1.x, x1.y, x1.z, 1 };

					float4 b;
					b.x = X1Inv[0].x * x1_float4.x + X1Inv[1].x * x1_float4.y + X1Inv[2].x * x1_float4.z + X1Inv[3].x * x1_float4.w;
					b.y = X1Inv[0].y * x1_float4.x + X1Inv[1].y * x1_float4.y + X1Inv[2].y * x1_float4.z + X1Inv[3].y * x1_float4.w;
					b.z = X1Inv[0].z * x1_float4.x + X1Inv[1].z * x1_float4.y + X1Inv[2].z * x1_float4.z + X1Inv[3].z * x1_float4.w;

					b.w = 1 - (b.x + b.y + b.z);
					//printf("%2f, %2f, %2f, %2f\n", b.x, b.y, b.z, b.w);
					if (OutsideTest(b))
						continue;
					// float4 inPos = X0*b;
					//vec3 inPos = A[0] * b.x + B[0] * b.y + C[0] * b.z + D[0] * b.w;
					vec3 inPos = { (A[0].x * b.x) + (B[0].x * b.y) + (C[0].x * b.z) + (D[0].x * b.w) , (A[0].y * b.x) + (B[0].y * b.y) + (C[0].y * b.z) + (D[0].y * b.w) ,(A[0].z * b.x) + (B[0].z * b.y) + (C[0].z * b.z) + (D[0].z * b.w) };
					/*inPos.x = (A[0].x * b.x) + (B[0].x * b.y) + (C[0].x * b.z) + (D[0].x * b.w);
					inPos.y = (A[0].y * b.x) + (B[0].y * b.y) + (C[0].y * b.z) + (D[0].y * b.w);
					inPos.z = (A[0].z * b.x) + (B[0].z * b.y) + (C[0].z * b.z) + (D[0].z * b.w);*/
					//printf("%2f, %2f, %2f\n", inPos.x, inPos.y, inPos.z);

					float4 den = tex3D<float4>(voltexObj, inPos.x, inPos.y, inPos.z);//*32767
					//printf("%2f, %2f, %2f,%2f\n", den.x, den.y, den.z, den.w);
					//float den2 = tex3D<float>(voltexObj, inPos.x, inPos.y, inPos.z);
					//float den3 = tex3D<float>(voltexObj, inPos.x, inPos.y, inPos.z);
					//float den4 = tex3D<float>(voltexObj, inPos.x, inPos.y, inPos.z);

					pOutGrid[z * 256 * 256 + y * 256 + x].x = den.x;
					pOutGrid[z * 256 * 256 + y * 256 + x].y = den.y;
					pOutGrid[z * 256 * 256 + y * 256 + x].z = den.z;
					pOutGrid[z * 256 * 256 + y * 256 + x].w = den.w;
					//printf("%2f, %2f, %2f,%2f\n", pOutGrid[z * 256 * 256 + y * 256 + x].x, pOutGrid[z * 256 * 256 + y * 256 + x].y, pOutGrid[z * 256 * 256 + y * 256 + x].z, pOutGrid[z * 256 * 256 + y * 256 + x].w);
					//pOutGrid[z * 256 * 256 + y * 256 + x].a() = den;//outgird에 1차원으로 표현한 배열에 해당 값을 할당
				}
			}
		}
	}
	__device__ vec3 getVertex(int x, int y, int z, Vertex d_vertices[]) {//바뀐 좌표를 구하는 함수
		// CUDA device variable to store the mode

		vec3 pos(x, y, z);

		if (device_mode == 0) { //Normal
			return pos;
		}

		else if (device_mode == 1) { //PBD
			vec3 pbd_pos;
			pbd_pos.x = pos.x / PBD_SCALE;
			pbd_pos.y = pos.y / PBD_SCALE;
			pbd_pos.z = pos.z / PBD_SCALE;

			//printf("%f, %f, %f\n", pbd_pos.x, pbd_pos.y, pbd_pos.z);
			int ix = int(pbd_pos.x); //0 ~ 16
			float wx = pbd_pos.x - ix; //
			int iy = int(pbd_pos.y); //0 ~ 16
			float wy = pbd_pos.y - iy;
			int iz = int(pbd_pos.z); // 0~9
			float wz = pbd_pos.z - iz; //

			vec3 origin_pos0(d_vertices[ix * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz].px, d_vertices[ix * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz].py, d_vertices[ix * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz].pz);
			vec3 origin_pos1(d_vertices[ix * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz + 1].px, d_vertices[ix * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz + 1].py, d_vertices[ix * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz + 1].pz);
			vec3 origin_pos2(d_vertices[ix * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz].px, d_vertices[ix * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz].py, d_vertices[ix * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz].pz);
			vec3 origin_pos3(d_vertices[ix * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz + 1].px, d_vertices[ix * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz + 1].py, d_vertices[ix * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz + 1].pz);
			vec3 origin_pos4(d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz].px, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz].py, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz].pz);
			vec3 origin_pos5(d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz + 1].px, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz + 1].py, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+iy * (PBD_Z)+iz + 1].pz);
			vec3 origin_pos6(d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz].px, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz].py, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz].pz);
			vec3 origin_pos7(d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz + 1].px, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz + 1].py, d_vertices[(ix + 1) * (PBD_Z) * (PBD_Y)+(iy + 1) * (PBD_Z)+iz + 1].pz);

			vec3 retval =
				(1 - wx) * (1 - wy) * (1 - wz) * origin_pos0 +
				(1 - wx) * (1 - wy) * wz * origin_pos1 +
				(1 - wx) * wy * (1 - wz) * origin_pos2 +
				(1 - wx) * wy * wz * origin_pos3 +
				wx * (1 - wy) * (1 - wz) * origin_pos4 +
				wx * (1 - wy) * wz * origin_pos5 +
				wx * wy * (1 - wz) * origin_pos6 +
				wx * wy * wz * origin_pos7;
			//printf("retval : %f, %f, %f\n", retval.x, retval.y, retval.z);

			retval.x = retval.x * PBD_SCALE;
			retval.y = retval.y * PBD_SCALE;
			retval.z = retval.z * PBD_SCALE;

			//printf("retval : %f, %f, %f\n", retval.x, retval.y, retval.z);
			return retval;
		}

		else if (device_mode == 2) { //Wave
			float delta = sinf(float(z * 0.03F) + mytime[0]);
			vec3 dis = vec3(delta * 20, 0, 0);
			vec3 retval = pos + dis; // 258 258 152
			//printf("%d\n", ttime);
			//256, 256, 150
			//Return vec3(x, y, z); // 임시로 변환 없음
			//Return retval;

			return retval;

		}

		else if (device_mode == 3) { //Bubble
			vec3 center(x / 2, y / 2, z / 2);
			vec3 dis = pos - center; /// 1 1 1
			float len = length(dis);
			float size = (sinf(mytime[0] * 0.05f)) * 40; // -1+1;

			//float size = sinf(10 * 0.05f) * 40; // -1+1;
			dis = dis * ((len + size) / len);
			vec3 retval = center + dis;
			return retval;
		}

		else if (device_mode == 4) { //Twist
			vec3 center(x / 2, y / 2, z / 2);
			vec3 dis = pos - center; /// 1 1 1
			dis.z = 0;
			float theta = sinf(mytime[0] * 0.1f) * (pos.z - center.z) * 0.01f * 1.3; // mytime[0] + pos.m[2]
			float c = cosf(theta);
			float s = sinf(theta);
			float dx = c * dis.x - s * dis.y;
			float dy = s * dis.x + c * dis.y;
			//dis.x = x;
			//dis.y = y;
			vec3 retval;
			retval.x = center.x + dx;
			retval.y = center.y + dy;
			retval.z = pos.z;
			return retval;
		}
	}

#define BX 4
#define BY 4
#define BZ 4
	__device__ static float maxValX = 0.0f;
	__device__ static float maxValY = 0.0f;
	__device__ static float maxValZ = 0.0f;
	__device__ static float minValX = 0.0f;
	__device__ static float minValY = 0.0f;
	__device__ static float minValZ = 0.0f;
	__global__ void resampleKernel(const uint32_t vol_size, float4* input, cudaTextureObject_t voltexObj, Vertex d_vertices[])
	{

		//thread 0~63 ,block 전체크기/64
		//int x, y, z;
		//x = blockIdx.x * blockDim.x + threadIdx.x; // thread :32->512
		//y = blockIdx.y * blockDim.y + threadIdx.y;
		//z = blockIdx.z * blockDim.z + threadIdx.z; // z의 범위를 조심. 배수.
		__shared__ vec3 buf[BZ + 1][BY + 1][BX + 1][2];

		int blockIdx_z = blockIdx.x / (64 * 64);
		int blockIdx_y = (blockIdx.x / 64) % 64;
		int blockIdx_x = blockIdx.x % 64;

		int myId = threadIdx.x; //전체에서의 위치
		const int a = (BX + 1) * (BY + 1);
		int tz = myId / a, ty = (myId / (BX + 1)) % (BY + 1), tx = myId % (BX + 1); //블럭 안에서의 위치

		buf[tz][ty][tx][0] = vec3(blockIdx_x * BX + tx, blockIdx_y * BY + ty, blockIdx_z * BZ + tz);
		buf[tz][ty][tx][1] = getVertex(blockIdx_x * BX + tx, blockIdx_y * BY + ty, blockIdx_z * BZ + tz, d_vertices);
		/*cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
		   printf( "resampleKernel buf failed: %s\n", cudaGetErrorString(cudaStatus));
		   return;
		}*/

		myId += BX * BY * BZ; // + 512 

		if (myId < (BX + 1) * (BY + 1) * (BZ + 1)) {
			tz = myId / a, ty = (myId / (BX + 1)) % (BY + 1), tx = myId % (BX + 1);

			buf[tz][ty][tx][0] = vec3(blockIdx_x * BX + tx, blockIdx_y * BY + ty, blockIdx_z * BZ + tz);
			buf[tz][ty][tx][1] = getVertex(blockIdx_x * BX + tx, blockIdx_y * BY + ty, blockIdx_z * BZ + tz, d_vertices);
			//maxValX = __max(maxValX, buf[tz][ty][tx][1].x);
			//maxValY = __max(maxValY, buf[tz][ty][tx][1].y);
			//maxValZ = __max(maxValZ, buf[tz][ty][tx][1].z);
			//minValX = __min(minValX, buf[tz][ty][tx][1].x);
			//minValY = __min(minValY, buf[tz][ty][tx][1].y);
			//minValZ = __min(minValZ, buf[tz][ty][tx][1].z);

			//printf("buf: %f, %f, %f\n", buf[tz][ty][tx][1].x, buf[tz][ty][tx][1].y, buf[tz][ty][tx][1].z);
			//printf("max: %f, %f, %f\n", maxValX, maxValY, maxValZ);
			//printf("min: %f, %f, %f\n", minValX, minValY, minValZ);
		}
		__syncthreads();
		//tx ty tz 다시 설정
		//tx = threadIdx.x / 4;
		//ty = threadIdx.y / 4;
		//tz = threadIdx.z / 4;

		tz = threadIdx.x / 64 * 64;
		ty = (threadIdx.x / 64) % 64;
		tx = threadIdx.x % 64;


		int forDebug = 0; // (x == 1 && y == 1 && z == 0) ? 1 : 0;
		SampleTetrahedron(buf[tz][ty][tx], buf[tz + 1][ty][tx], buf[tz + 1][ty][tx + 1], buf[tz + 1][ty + 1][tx], input, voltexObj, forDebug);//사면체를 생성
		SampleTetrahedron(buf[tz + 1][ty + 1][tx], buf[tz + 1][ty + 1][tx + 1], buf[tz + 1][ty][tx + 1], buf[tz][ty + 1][tx + 1], input, voltexObj, forDebug);
		SampleTetrahedron(buf[tz + 1][ty][tx + 1], buf[tz][ty][tx + 1], buf[tz][ty][tx], buf[tz][ty + 1][tx + 1], input, voltexObj, forDebug);
		SampleTetrahedron(buf[tz][ty][tx], buf[tz][ty + 1][tx], buf[tz + 1][ty + 1][tx], buf[tz][ty + 1][tx + 1], input, voltexObj, forDebug);
		SampleTetrahedron(buf[tz + 1][ty + 1][tx], buf[tz + 1][ty][tx + 1], buf[tz][ty + 1][tx + 1], buf[tz][ty][tx], input, voltexObj, forDebug);

	}

	__global__ void brightenImageKernel(unsigned char* outImage, cudaTextureObject_t voltexObj, float zoom)
	{
		//printf("brightenImage test\n");	
		int ix = threadIdx.x + blockIdx.x * 16;
		int iy = threadIdx.y + blockIdx.y * 16;

		//int blockIdx_z = blockIdx.x / (64 * 64);
		//int blockIdx_y = (blockIdx.x / 64) % 64;
		//int blockIdx_x = blockIdx.x % 64;
		//int myId = threadIdx.x;
		int offset = (iy * Window_WIDTH + ix) * 3;//현재 위치

		float asum = 0, rsum = 0, gsum = 0, bsum = 0;

		vec3 start;
		start.x = deye[0].x + duvw[0].x * ((ix - Window_WIDTH * 0.5)) * zoom + duvw[1].x * ((iy - Window_HEIGHT * 0.5)) * zoom;
		start.y = deye[0].y + duvw[0].y * ((ix - Window_WIDTH * 0.5)) * zoom + duvw[1].y * ((iy - Window_HEIGHT * 0.5)) * zoom;
		start.z = deye[0].z + duvw[0].z * ((ix - Window_WIDTH * 0.5)) * zoom + duvw[1].z * ((iy - Window_HEIGHT * 0.5)) * zoom;
		float trange[2];
		RBInter(start, trange);//raystart
		trange[0] += 0.01f;
		trange[1] -= 0.01f;

		for (float t = trange[0]; t < trange[1]; t += 1.0) {

			vec3 sample;//rayStart 파트

			sample.x = start.x + duvw[2].x * t;//rayStart 파트
			sample.y = start.y + duvw[2].y * t;//rayStart 파트
			sample.z = start.z + duvw[2].z * t;//rayStart 파트


			float4 texValue = tex3D<float4>(voltexObj, sample.x + 0.5, sample.y + 0.5, sample.z + 0.5);


			//float alpha = 1.0 - powf(2.71828, -(texValue.w));

			float alpha = texValue.w;
			//if (alpha <= 20 || alpha >= 40) {
			//	alpha = 0;
			//}
			//
			//alpha = 1.0f - pow(2.71828, -(alpha - 1.0f));

			vec3 color;
			//= vec3(255.0f * texValue.x, 255.0f * texValue.y, 255.0f * texValue.z);
			color.x = 255.0f * texValue.x;
			color.y = 255.0f * texValue.y;
			color.z = 255.0f * texValue.z;


			if (alpha < 0)
				continue;

			//float Ia = 0.4, Id = 0.6, Is = 0.7;
			//color = CalcLight(sample, voltexObj, color, Ia, Id, Is);

			rsum = 255.0;// rsum + alpha * color.x * (1 - asum);
			gsum = gsum + alpha * color.y * (1 - asum);
			bsum = bsum + alpha * color.z * (1 - asum);
			asum = asum + alpha * (1 - asum);

			if (asum > 0.99)
				break;

		}

		outImage[offset] = __min((int)(rsum), 255);
		outImage[offset + 1] = __min((int)(gsum), 255);
		outImage[offset + 2] = __min((int)(bsum), 255);
	}
	cudaError_t Testbed::addWithCuda(unsigned char* out)
	{
		cudaError_t cudaStatus;
		float elapsed_time_ms = 0;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		cudaMemset(dev_input, 0, VOLX * VOLY * VOLZ * sizeof(float4));
		//dim3 b(VOLX / 8, VOLY / 8, VOLZ / 8);// VOLY, VOLZ, 1);//블록
		dim3 b(VOLX / 8, VOLY / 8, VOLZ / 8);// VOLY, VOLZ, 1);//블록
		//dim3 t(8, 8, 8);// (VOLX/64, 8, 8) //스레드
		dim3 t(8, 8, 8);// (VOLX/64, 8, 8) //스레드0~63
		//resampleKernel << <b, t >> > (dev_input, voltexObj);


		my_linear_kernel(resampleKernel, 3000, m_stream.get(),
			256 * 256 * 256,
			dev_input,
			voltexObj,
			d_vertices
		);


		//linear_kernel(resampleKernel, 0, m_stream.get(),
		//	3000,
		//	dev_input,
		//	voltexObj
		//);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "resampleKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return cudaStatus;
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		elapsed_time_ms = 0;
		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		//printf("1.resampling time: %8.2f ms\n", elapsed_time_ms);

		cudaExtent OutVolumeExtent = { VOLX,VOLY,VOLZ };
		cudaMemcpy3DParms param = { 0 };
		param.srcPtr = make_cudaPitchedPtr(dev_input, 256 * sizeof(float4), 256, 256);
		param.dstArray = pDevOutArr;
		param.extent = OutVolumeExtent;
		param.kind = cudaMemcpyDeviceToDevice;
		cudaError_t err = cudaMemcpy3D(&param);

		if (err != cudaSuccess)
		{
			printf("Error3: cudaMemcpy3D(pDevOutVolume) - (error: %s)\n", cudaGetErrorString(err));
			return err;
		}

		initTexture(&voltexOutObj, pDevOutArr);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("resampleKernel initTexture failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		cudaEventRecord(start, 0);

		dim3 b1(32, 32, 1);
		dim3 t1(16, 16, 1);
		brightenImageKernel << <b1, t1 >> > (dev_output, voltexOutObj, zoom);
		//my_linear_kernel(brightenImageKernel, 3000, m_stream.get(),
		//	256 * 256 * 256,
		//	dev_output,
		//	voltexOutObj
		//);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("brightenImageKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize return  failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		elapsed_time_ms = 0;
		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		//printf("2.rendering time: %8.2f ms\n", elapsed_time_ms);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaStatus = cudaMemcpy(out, dev_output, Window_WIDTH * Window_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed!9: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}


		return cudaStatus;
	}
	//cudaError_t Testbed::addWithCuda2(unsigned char* out)
	//{
	//	cudaError_t cudaStatus;
	//	float elapsed_time_ms = 0;

	//	cudaEvent_t start, stop;
	//	cudaEventCreate(&start);
	//	cudaEventCreate(&stop);

	//	cudaEventRecord(start, 0);

	//	dim3 b1(32, 32, 1);
	//	dim3 t1(16, 16, 1);
	//	brightenImageKernel << <b1, t1 >> > (dev_output, voltexOutObj);

	//	// Check for any errors launching the kernel
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		printf("brightenImageKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//		return cudaStatus;
	//	}

	//	cudaStatus = cudaDeviceSynchronize();
	//	if (cudaStatus != cudaSuccess) {
	//		printf("cudaDeviceSynchronize return  failed: %s\n", cudaGetErrorString(cudaStatus));
	//		return cudaStatus;
	//	}
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	elapsed_time_ms = 0;
	//	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	//	printf("2.rendering time: %8.2f ms\n", elapsed_time_ms);

	//	cudaEventDestroy(start);
	//	cudaEventDestroy(stop);
	//	cudaStatus = cudaMemcpy(out, dev_output, 512 * 512 * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//	if (cudaStatus != cudaSuccess) {
	//		printf("cudaMemcpy failed!9: %s\n", cudaGetErrorString(cudaStatus));
	//		return cudaStatus;
	//	}
	//	return cudaStatus;
	//}

	//cudaError_t Testbed::addWithCuda3(unsigned char* out){
	//	cudaError_t cudaStatus = addWithCuda((unsigned char*)out);
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stderr, "cudamain addWithCuda failed!: %s\n", cudaGetErrorString(cudaStatus));
	//		return cudaStatus;
	//	}
	//	/* cudaStatus = addWithCuda2((unsigned char*)out);
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stderr, "cudamain addWithCuda2 failed!: %s\n", cudaGetErrorString(cudaStatus));
	//		return cudaStatus;
	//	}*/
	//}
	cudaArray* Testbed::mallocArr(vec4* pVol, cudaExtent volumeExtent) {

		// gpu memory alloc
		cudaError_t cudaStatus;

		//  cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "mallocArray createChannelDesc failed!- (error: %s)\n", cudaGetErrorString(cudaStatus));
		}

		cudaError_t err = cudaMalloc3DArray(&pDevArr, &channelDesc, volumeExtent);
		if (err != cudaSuccess)
		{
			printf("Error1: cudaMemcpy3D(pDevVolume) - (error: %s)\n", cudaGetErrorString(err));
			return nullptr;
		}
		// memory copy host to device
		cudaMemcpy3DParms param = { 0 };
		param.srcPtr = make_cudaPitchedPtr(pVol, VOLX * sizeof(float4), VOLX, VOLZ);
		param.dstArray = pDevArr;
		param.extent = volumeExtent;
		param.kind = cudaMemcpyDeviceToDevice;
		err = cudaMemcpy3D(&param);
		if (err != cudaSuccess)
		{
			printf("Error2: cudaMemcpy3D(pDevVolume) - (error: %s)\n", cudaGetErrorString(err));
			return nullptr;
		}
		return pDevArr;
	}

	bool Testbed::initTexture(cudaTextureObject_t* pTex, cudaArray* pDevVol) {

		cudaError_t cudaStatus;
		// 입력 gpu array를 리소스 형식으로 해석Get
		cudaResourceDesc            texRes;//메모리의 종류
		memset(&texRes, 0, sizeof(cudaResourceDesc));//메모리 초기화
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = pDevVol;//cudaArray

		// texture를 처리하는 방법을 정의
		cudaTextureDesc texDescr; // = { 0 }; 이것으로 안됨
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = false; // access with normalized texture coordinates 텍스처 좌표 0~1이 기본값인데 0~255로(원래 데이터 사이즈) 접근할 수 있도록하는 
		texDescr.filterMode = cudaFilterModeLinear; //cudaReadModeElementType&cudaFilterModeLiner 조합은 linear interpolation 잠정적 문제 발생 가능성 있음

		// wrap texture coordinates
		texDescr.addressMode[0] = cudaAddressModeBorder;
		texDescr.addressMode[1] = cudaAddressModeBorder;
		texDescr.addressMode[2] = cudaAddressModeBorder;
		texDescr.borderColor[0] = texDescr.borderColor[1] = texDescr.borderColor[2] = texDescr.borderColor[3] = 0;//텍스처 좌표 빠져나갈때 예외처리
		texDescr.readMode = cudaReadModeElementType; //cudaReadModeNormalizedFloat  보간을 수행할때 필수 0~1 범위값임

		// texture 생성
		cudaCreateTextureObject(pTex, &texRes, &texDescr, NULL);//(리턴값,구조체, 텍스쳐전용 정보
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "initTexture createTextureObject failed! - (error: %s)\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return true;
	}

	cudaError_t Testbed::InitData()
	{
		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			return cudaStatus;
		}
		//out은 WIDTH*HEIGHT 크기
		cudaStatus = cudaMalloc((void**)&dev_output, 3 * Window_WIDTH * Window_HEIGHT * sizeof(unsigned char)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_input, VOLX * VOLY * VOLZ * sizeof(float4)); //gpu 메모리를 size만큼 잡아주는 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(Testbed::volume, Testbed::rgba.data(), 256 * 256 * 256 * sizeof(float4), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!6");
			return cudaStatus;
		}
		cudaExtent volumeExtent = { 256,256,256 }; // w, h, d;
		//cudaExtent volumeExtent = { VOLX*4,VOLY*4,VOLZ*4 };

		pDevArr = mallocArr(Testbed::rgba.data(), volumeExtent); // gpu 메모리 할당과 복사 수행됨
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "initdata mallocArr failed!- (error: %s)\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		initTexture(&voltexObj, pDevArr);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "initdata initTexture failed!- (error: %s)\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "mallocArray createChannelDesc failed!- (error: %s)\n", cudaGetErrorString(cudaStatus));
		}

		cudaError_t err = cudaMalloc3DArray(&pDevOutArr, &channelDesc, volumeExtent);
		if (err != cudaSuccess)
		{
			printf("pDevOutArr: cudaMemcpy3D(pDevVolume) - (error: %s)\n", cudaGetErrorString(err));
			return err;
		}

		//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "mallocArray createChannelDesc failed!- (error: %s)\n", cudaGetErrorString(cudaStatus));
		//}
		//cudaExtent OutVolumeExtent = { 256,256,256 };

		////  cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);
		//

		//cudaStatus = cudaMalloc3DArray(&pDevOutArr, &channelDesc, OutVolumeExtent);
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "Error: cudaMalloc3DArray(pDevOutVolume) - (error: %s)\n", cudaGetErrorString(cudaStatus));
		//	return cudaStatus;
		//}
		return cudaStatus;
	}

	int Testbed::cudamain(float eye[3])
	{
		static float t = 0;
		t += 4.0;//시간측정
		vecsm heye(eye[0], eye[1], eye[2]);//host 변수

		cudaMemcpyToSymbol(mytime, &t, sizeof(float));
		vecsm uvw[3];
		vecsm up(0, -1, 0);
		vecsm hat(VOLX * 0.5, VOLY * 0.5, VOLZ * 0.5);
		uvw[2] = hat - heye;//cameradir
		uvw[2].make_unit_vector();
		uvw[0] = up.CrossProduct(uvw[2]); //cameraright
		uvw[0].make_unit_vector();
		uvw[1] = uvw[2].CrossProduct(uvw[0]);//cameranewup
		uvw[1].make_unit_vector();

		cudaMemcpyToSymbol(deye, heye.m, sizeof(float3));//memcpy 
		cudaMemcpyToSymbol(duvw, uvw, 3 * sizeof(float3));

		cudaError_t cudaStatus = addWithCuda((unsigned char*)MyTexture);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamain addWithCuda failed!: %s\n", cudaGetErrorString(cudaStatus));
			return 1;
		}

		return 0;
	}

	void Testbed::initPBD() {
		printf("merging start\n");

		// init in CPU is perfectly done!!!! gauranteed
		Testbed::InitVertices();
		printf("merging InitVertices\n");

		Testbed::InitEdge();
		printf("merging initEdge\n");

		//InitStiffness();
		Testbed::InitParallelizableGroups();//initParallelizableEdgeGroups
		printf("merging InitParallelizableEdgeGroups\n");

		int DGroupMax_size = 0;
		for (int i = 0; i < g_prallelizableEdgeGroups.size(); i++) {
			DGroupMax_size = __max(DGroupMax_size, g_prallelizableEdgeGroups[i].size());
		}
		int VGroupMax_size = 0;
		for (int i = 0; i < g_prallelizableTetGroups.size(); i++) {
			VGroupMax_size = __max(VGroupMax_size, g_prallelizableTetGroups[i].size());
		}
		//thth
		int SGroupMax_size = 0;
		for (int i = 0; i < g_prallelizableSkinGroups.size(); i++) {
			SGroupMax_size = __max(SGroupMax_size, g_prallelizableSkinGroups[i].size());
		}
		printf("merging PBDGPUMemoryAlloc\n");
		printf("DGroupMax_size : %d \t", DGroupMax_size);
		printf("VGroupMax_size : %d \t", VGroupMax_size);
		printf("SGroupMax_size : %d \t", SGroupMax_size);

		//여기 DGroupMax_size가 아니라 VGroupMax_size아닌가요?? - thth
		Testbed::PBDGPUMemoryAlloc(g_edges.size(), DGroupMax_size, g_tets.size(), VGroupMax_size, g_skins.size(), SGroupMax_size);

		//PBDGPUMemoryAlloc(g_edges.size()); // not gauranteed

		printf("merging end\n");
		////////////////////////////////////////////////////////////////////
		printf("PBD Sclae : %d * %d * %d cellSize\n", PBD_X, PBD_Y, PBD_Z);

		PBD_flag = false;
	}
	// Function to copy the selected option to the CUDA device
	void Testbed::saveOptionToCuda(int option) {
		// Copy the selected option index to the CUDA device variable
		cudaMemcpyToSymbol(device_mode, &option, sizeof(int));
		if (option == 1) {
			printf("PBD Mode\n");
		}
		else if (option == 2) {
			printf("Wave Mode\n");
		}
		else if (option == 3) {
			printf("Bubble Mode\n");
		}
		else if (option == 4) {
			printf("Twist Mode\n");
		}
		else printf("Normal Mode\n");
	}

	// 카메라 설정
	glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
	glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

	glm::vec2 windowToNDC(const glm::vec2& windowCoord, int windowWidth, int windowHeight) {
		float x = 2.0f * (windowCoord.x / windowWidth) - 1.0f;
		float y = 1.0f - 2.0f * (windowCoord.y / windowHeight);
		return glm::vec2(x, y);
	}

	glm::vec3 screenToWorldRay(const glm::vec2& screenCoord, const glm::mat4& projection, const glm::mat4& view) {
		// NDC로 변환
		glm::vec4 clipCoords(screenCoord.x, screenCoord.y, -1.0f, 1.0f);

		// 뷰 공간으로 변환
		glm::mat4 invProjection = glm::inverse(projection);
		glm::vec4 eyeCoords = invProjection * clipCoords;
		eyeCoords = glm::vec4(eyeCoords.x, eyeCoords.y, -1.0f, 0.0f);

		// 월드 공간으로 변환
		glm::mat4 invView = glm::inverse(view);
		glm::vec3 rayWorld = glm::vec3(invView * eyeCoords);
		rayWorld = glm::normalize(rayWorld);

		return rayWorld;
	}

	void Testbed::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			if (action == GLFW_PRESS) {
				mousePressed = true;
				glfwGetCursorPos(window, &lastX, &lastY);
			}
			else if (action == GLFW_RELEASE) {
				mousePressed = false;
			}
		}
		if (button == GLFW_MOUSE_BUTTON_RIGHT) {
			if (action == GLFW_PRESS) {
				glm::vec2 windowCoord(xpos, ypos);
				glm::vec2 ndcCoord = windowToNDC(windowCoord, WIDTH, HEIGHT);

				// 프로젝션 행렬과 뷰 행렬 설정
				glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
				glm::mat4 view = glm::lookAt(cameraPos, cameraFront + cameraPos, cameraUp);

				// 화면 좌표를 월드 공간 레이로 변환
				glm::vec3 rayWorld = screenToWorldRay(ndcCoord, projection, view);

				std::cout << "Ray in World Coordinates: (" << rayWorld.x << ", " << rayWorld.y << ", " << rayWorld.z << ")\n";
			}
			else if (action == GLFW_RELEASE) {
				mousePressed = false;
			}
		}
	}
#define PI 3.14159265358979323846

	void Testbed::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
		if (mousePressed) {
			float xoffset = xpos - lastX;
			float yoffset = ypos - lastY;
			lastX = xpos;
			lastY = ypos;

			float sensitivity = 0.1f;
			xoffset *= sensitivity;
			yoffset *= sensitivity;

			yaw += xoffset;
			pitch += yoffset;

			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;

			float radius = sqrt(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]);
			eye[0] = radius * cos(pitch * PI / 180.0f) * cos(yaw * PI / 180.0f);
			eye[1] = radius * sin(pitch * PI / 180.0f);
			eye[2] = radius * cos(pitch * PI / 180.0f) * sin(yaw * PI / 180.0f);
		}
	}
	void Testbed::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
		zoom -= yoffset * 0.1f;
		if (zoom < 0.1f) zoom = 0.1f; // 최소 줌 레벨 설정

	}
	void Testbed::updateCameraPosition() {
		float radius = zoom * 500.0f; // 초기 거리 값을 사용하여 조정
		eye[0] = radius * cos(pitch * PI / 180.0) * cos(yaw * PI / 180.0);
		eye[1] = radius * sin(pitch * PI / 180.0);
		eye[2] = radius * cos(pitch * PI / 180.0) * sin(yaw * PI / 180.0);
	}

	void Testbed::openPBDWindow(int option) {//지금 다른 파일에서 (testbed.cu) 파라미터가 vec4로 정의되어있는 부분이 있음

		LARGE_INTEGER Frequency;
		LARGE_INTEGER BeginTime;
		LARGE_INTEGER Endtime;

		saveOptionToCuda(option);

		//float eye[3] = { 200,200,500 };
		if (!glfwInit()) {
			std::cout << "GLFW를 초기화할 수 없습니다." << std::endl;
			return;
		}
		window = glfwCreateWindow(1024, 1024, "Title", NULL, NULL);
		if (!window) {
			std::cout << "GLFW 윈도우를 생성할 수 없습니다." << std::endl;
			glfwTerminate();
			return;
		}
		glfwMakeContextCurrent(window);
		glfwSwapInterval(1); // Enable vsync

		glfwSetWindowUserPointer(window, this);
		glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
			Testbed* tb = static_cast<Testbed*>(glfwGetWindowUserPointer(window));
			tb->mouse_button_callback(window, button, action, mods);
			});
		glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {
			Testbed* tb = static_cast<Testbed*>(glfwGetWindowUserPointer(window));
			tb->cursor_position_callback(window, xpos, ypos);
			});
		glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
			Testbed* tb = static_cast<Testbed*>(glfwGetWindowUserPointer(window));
			tb->scrollCallback(window, xoffset, yoffset);
			});


		while (!glfwWindowShouldClose(window)) {
			//printf("Start testwhile\n");
			// 화면 지우기
			glClear(GL_COLOR_BUFFER_BIT);
			// 텍스처 출력
			generateGradient();
			glfwSwapBuffers(window);
			// 이벤트 처리
			glfwPollEvents();

			//float eye[3] = { 200,200,500 };
			if (option == 1) {
				Update();
				QueryPerformanceFrequency(&Frequency);
				QueryPerformanceCounter(&BeginTime);
				CPtoGPU();
				for (int i = 0; i < 5; i++)
					Projection();
				GPUtoCP();
				QueryPerformanceCounter(&Endtime);
				long long elapsed = Endtime.QuadPart - BeginTime.QuadPart;
				double duringtime = (double)elapsed / (double)Frequency.QuadPart;
				printf("GPU TIME : %.8f\n", duringtime);
				velocityUpdate();
				CPtoGPU();
			}
			cudamain(eye);
			//printf("End testwhile\n");
		}
		// 리소스 정리
		glDeleteTextures(1, &texID);
		// GLFW 종료
		glfwDestroyWindow(window);
		glfwMakeContextCurrent(m_glfw_window);
	}

	//cuda 이미지 뽑아오기
	GPUMemory<vec4> Testbed::get_rgba_on_grid(ivec3 res3d, vec3 ray_dir, bool voxel_centers, float depth, bool density_as_alpha) {
		const uint32_t n_elements = (res3d.x * res3d.y * res3d.z);
		GPUMemory<vec4> rgba(n_elements);

		const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(m_stream.get());

		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
		const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);

		GPUMemory<float> positions(n_elements * floats_per_coord);

		const uint32_t batch_size = std::min(n_elements, 1u << 20);

		// generate inputs
		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res3d.x, threads.x), div_round_up((uint32_t)res3d.y, threads.y), div_round_up((uint32_t)res3d.z, threads.z) };
		generate_grid_samples_nerf_uniform_dir << <blocks, threads, 0, m_stream.get() >> > (
			//여기서 쿠다커널로 픽셀값 가져옴
			res3d,
			m_nerf.density_grid_ema_step,
			m_render_aabb,
			m_render_aabb_to_local,
			m_aabb,
			ray_dir,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)positions.data(), 1, 0, extra_stride),
			extra_dims_gpu,
			voxel_centers
			);

		// Only process 1m elements at a time
		for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
			uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

			// run network
			GPUMatrix<float> positions_matrix((float*)(positions.data() + offset * floats_per_coord), floats_per_coord, local_batch_size);
			GPUMatrix<float> rgbsigma_matrix((float*)(rgba.data() + offset), 4, local_batch_size);
			m_network->inference(m_stream.get(), positions_matrix, rgbsigma_matrix);
			linear_kernel(
				filter_with_occupancy,
				0,
				m_stream.get(),
				local_batch_size,
				positions_matrix.data(),
				floats_per_coord,
				m_nerf.density_grid_bitfield.data(),
				rgbsigma_matrix.data());

			// convert network output to RGBA (in place)
			//vec4 형 rgba 에 맞춰서 다시 데이터 넣어주고 반환함
			linear_kernel(compute_nerf_rgba_kernel,
				0,
				m_stream.get(),
				local_batch_size,

				rgba.data() + offset,
				m_nerf.rgb_activation,
				m_nerf.density_activation,
				depth,
				density_as_alpha);
			//printf(" offset : %d\n", offset);
			//post_process_volume(rgba.data() + offset, local_batch_size);
		}
		//std::vector<vec4> rgba_cpu;
		//rgba_cpu.resize(rgba.size());
		//rgba.copy_to_host(rgba_cpu);
		//float maxr = 0;
		//float maxg = 0;
		//float maxb = 0;
		//float maxa = 0;
		//const int volume_dim = 256;
		//const int margin = 70; // 가장자리에서 보존할 범위
		//for (size_t i = 0; i < rgba_cpu.size(); ++i) {
		//	// 볼륨 데이터의 좌표를 계산
		//	int ix = i % volume_dim;
		//	int iy = (i / volume_dim) % volume_dim;
		//	int iz = i / (volume_dim * volume_dim);

		//	maxr = __max(maxr, rgba_cpu[i].x);
		//	maxg = __max(maxg, rgba_cpu[i].y);
		//	maxb = __max(maxb, rgba_cpu[i].z);
		//	maxa = __max(maxa, rgba_cpu[i].w);

		//	// 중심에서 margin 범위 이내에 있는 부분은 보존
		//	if (ix >= volume_dim / 2 - margin && ix < volume_dim / 2 + margin &&
		//		iy >= volume_dim / 2 - margin && iy < volume_dim / 2 + margin + 20 &&
		//		iz >= volume_dim / 2 - margin && iz < volume_dim / 2 + margin) {
		//		continue; // 보존할 부분은 건너뜀
		//	}

		//	// 나머지 가장자리 부분을 투명하게 처리 (alpha 값을 0으로 설정)
		//	rgba_cpu[i].w = 0.0f;
		//}
		//const float density_threshold =5; // 임의의 값으로 설정
		//printf(" density_threshold : %f\n", density_threshold);
		//for (uint32_t i = 0; i < rgba_cpu.size(); ++i) {
		//	// 볼륨 데이터의 밀도가 임계값 미만인 경우 투명하게 처리 (불필요한 장애물 제거)
		//	if (rgba_cpu[i].w < density_threshold) {
		//		rgba_cpu[i].w = 0.0f; // 투명 처리 (또는 다른 처리 방법을 선택)
		//	}
		//}

		//printf("fun get_rgba_on_grid max r:!! %f !! \n", maxr);
		//printf("fun get_rgba_on_grid max g:!! %f !! \n", maxg);
		//printf("fun get_rgba_on_grid max b:!! %f !! \n", maxb);
		//printf("fun get_rgba_on_grid max a:!! %f !! \n", maxa);

		//vec4* rgbaData = rgba_cpu.data();
		//const int dataSize = 256 * 256 * 256;

		//// Open a binary file for writing
		//std::ofstream outFile("fox_rgba_data.den", std::ios::binary);

		//// Check if the file is opened successfully
		//if (!outFile.is_open()) {
		//	std::cerr << "Failed to open the file for writing." << std::endl;
		//	return 1;
		//}

		//// Write the data to the file
		//outFile.write(reinterpret_cast<char*>(rgbaData), dataSize * sizeof(float) * 4);

		//// Close the file
		//outFile.close();
		return rgba;
	}

	int Testbed::marching_cubes(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local, float thresh) {
		res3d.x = next_multiple((unsigned int)res3d.x, 16u);
		res3d.y = next_multiple((unsigned int)res3d.y, 16u);
		res3d.z = next_multiple((unsigned int)res3d.z, 16u);

		if (thresh == std::numeric_limits<float>::max()) {
			thresh = m_mesh.thresh;
		}

		GPUMemory<float> density = get_density_on_grid(res3d, aabb, render_aabb_to_local);
		marching_cubes_gpu(m_stream.get(), aabb, render_aabb_to_local, res3d, thresh, density, m_mesh.verts, m_mesh.indices);

		uint32_t n_verts = (uint32_t)m_mesh.verts.size();
		m_mesh.verts_gradient.resize(n_verts);

		m_mesh.trainable_verts = std::make_shared<TrainableBuffer<3, 1, float>>(std::array<int, 1>{ {(int)n_verts}});
		m_mesh.verts_gradient.copy_from_device(m_mesh.verts); // Make sure the vertices don't get destroyed in the initialization

		pcg32 rnd{ m_seed };
		m_mesh.trainable_verts->initialize_params(rnd, (float*)m_mesh.verts.data());
		m_mesh.trainable_verts->set_params((float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
		m_mesh.verts.copy_from_device(m_mesh.verts_gradient);

		m_mesh.verts_optimizer.reset(create_optimizer<float>({
			{"otype", "Adam"},
			{"learning_rate", 1e-4},
			{"beta1", 0.9f},
			{"beta2", 0.99f},
			}));

		m_mesh.verts_optimizer->allocate(m_mesh.trainable_verts);

		compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);
		compute_mesh_vertex_colors();

		return (int)(m_mesh.indices.size() / 3);
	}

	uint8_t* Testbed::Nerf::get_density_grid_bitfield_mip(uint32_t mip) {
		return density_grid_bitfield.data() + grid_mip_offset(mip) / 8;
	}

	void Testbed::Nerf::reset_extra_dims(default_rng_t& rng) {
		uint32_t n_extra_dims = training.dataset.n_extra_dims();
		std::vector<float> extra_dims_cpu(n_extra_dims * (training.dataset.n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
		float* dst = extra_dims_cpu.data();
		training.extra_dims_opt = std::vector<VarAdamOptimizer>(training.dataset.n_images, VarAdamOptimizer(n_extra_dims, 1e-4f));
		for (uint32_t i = 0; i < training.dataset.n_images; ++i) {
			vec3 light_dir = warp_direction(normalize(training.dataset.metadata[i].light_dir));
			training.extra_dims_opt[i].reset_state();
			std::vector<float>& optimzer_value = training.extra_dims_opt[i].variable();
			for (uint32_t j = 0; j < n_extra_dims; ++j) {
				if (training.dataset.has_light_dirs && j < 3) {
					dst[j] = light_dir[j];
				}
				else {
					dst[j] = random_val(rng) * 2.0f - 1.0f;
				}
				optimzer_value[j] = dst[j];
			}
			dst += n_extra_dims;
		}
		training.extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);

		rendering_extra_dims.resize(training.dataset.n_extra_dims());
		CUDA_CHECK_THROW(cudaMemcpy(rendering_extra_dims.data(), training.extra_dims_gpu.data(), rendering_extra_dims.bytes(), cudaMemcpyDeviceToDevice));
	}

	const float* Testbed::Nerf::get_rendering_extra_dims(cudaStream_t stream) const {
		CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

		if (training.dataset.n_extra_dims() == 0) {
			return nullptr;
		}

		const float* extra_dims_src = rendering_extra_dims_from_training_view >= 0 ?
			training.extra_dims_gpu.data() + rendering_extra_dims_from_training_view * training.dataset.n_extra_dims() :
			rendering_extra_dims.data();

		if (!training.dataset.has_light_dirs) {
			return extra_dims_src;
		}

		// the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
		// we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
		size_t size = training.dataset.n_extra_dims() * sizeof(float);
		float* dims_gpu = training.extra_dims_gpu.data() + training.dataset.n_images * training.dataset.n_extra_dims();
		CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
		vec3 light_dir = warp_direction(normalize(light_dir));
		CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(vec3)), cudaMemcpyHostToDevice, stream));
		return dims_gpu;
	}

	int Testbed::Nerf::find_closest_training_view(mat4x3 pose) const {
		int bestimage = training.view;
		float bestscore = std::numeric_limits<float>::infinity();
		for (int i = 0; i < training.n_images_for_training; ++i) {
			float score = distance(training.transforms[i].start[3], pose[3]);
			score += 0.25f * distance(training.transforms[i].start[2], pose[2]);
			if (score < bestscore) {
				bestscore = score;
				bestimage = i;
			}
		}

		return bestimage;
	}

	void Testbed::Nerf::set_rendering_extra_dims_from_training_view(int trainview) {
		if (!training.dataset.n_extra_dims()) {
			throw std::runtime_error{ "Dataset does not have extra dims." };
		}

		if (trainview < 0 || trainview >= training.dataset.n_images) {
			throw std::runtime_error{ "Invalid training view." };
		}

		rendering_extra_dims_from_training_view = trainview;
	}

	void Testbed::Nerf::set_rendering_extra_dims(const std::vector<float>& vals) {
		CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

		if (vals.size() != training.dataset.n_extra_dims()) {
			throw std::runtime_error{ fmt::format("Invalid number of extra dims. Got {} but must be {}.", vals.size(), training.dataset.n_extra_dims()) };
		}

		rendering_extra_dims_from_training_view = -1;
		rendering_extra_dims.copy_from_host(vals);
	}

	std::vector<float> Testbed::Nerf::get_rendering_extra_dims_cpu() const {
		CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

		if (training.dataset.n_extra_dims() == 0) {
			return {};
		}

		std::vector<float> extra_dims_cpu(training.dataset.n_extra_dims());
		CUDA_CHECK_THROW(cudaMemcpy(extra_dims_cpu.data(), get_rendering_extra_dims(nullptr), rendering_extra_dims.bytes(), cudaMemcpyDeviceToHost));

		return extra_dims_cpu;
	}

}

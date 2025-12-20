/**
 * Activation tracker implementation
 */

#include "activation_tracker.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <limits>
#include <tuple>

// Simple image loading using stb_image (header-only, no external deps needed)
// For now, we'll use a simple approach - require images to be preprocessed
// or use a minimal image loader
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace darknet {

ActivationTracker::ActivationTracker(const CfgParser& cfg_parser, bool verbose)
    : cfg_parser_(cfg_parser), verbose_(verbose) {
    auto layers = cfg_parser_.get_layers();
    layer_stats_.resize(layers.size());
}

ActivationTracker::~ActivationTracker() {
}

std::vector<float> ActivationTracker::load_and_preprocess_image(
    const std::string& image_path,
    int target_w,
    int target_h
) {
    int img_w, img_h, img_c;
    unsigned char* img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
    
    if (!img_data) {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return std::vector<float>();
    }
    
    if (img_c != 3) {
        std::cerr << "Error: Image must have 3 channels (RGB), got " << img_c << std::endl;
        stbi_image_free(img_data);
        return std::vector<float>();
    }
    
    // Letterbox: resize with aspect ratio preservation and padding
    float scale = std::min(static_cast<float>(target_w) / img_w, 
                          static_cast<float>(target_h) / img_h);
    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);
    
    // Create output buffer (target_w x target_h x 3)
    std::vector<float> output(target_w * target_h * 3, 0.5f);  // Gray padding (0.5 normalized)
    
    // Resize and place image in center
    int pad_x = (target_w - new_w) / 2;
    int pad_y = (target_h - new_h) / 2;
    
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = static_cast<int>(x / scale);
            int src_y = static_cast<int>(y / scale);
            
            if (src_x < img_w && src_y < img_h) {
                int src_idx = (src_y * img_w + src_x) * 3;
                int dst_idx = ((pad_y + y) * target_w + (pad_x + x)) * 3;
                
                // Normalize to [0, 1] and convert BGR to RGB if needed
                output[dst_idx + 0] = img_data[src_idx + 2] / 255.0f;  // R
                output[dst_idx + 1] = img_data[src_idx + 1] / 255.0f;  // G
                output[dst_idx + 2] = img_data[src_idx + 0] / 255.0f;  // B
            }
        }
    }
    
    stbi_image_free(img_data);
    return output;
}

std::vector<int32_t> ActivationTracker::compute_activation_q_values(
    const std::vector<std::string>& calibration_images,
    const std::string& weights_path,
    BatchNormFolder& bn_folder,
    Int16Quantizer& quantizer
) {
    if (calibration_images.empty()) {
        std::cerr << "Error: No calibration images provided" << std::endl;
        return std::vector<int32_t>();
    }
    
    auto layers = cfg_parser_.get_layers();
    auto conv_layers = cfg_parser_.get_conv_layers();
    const NetworkConfig& net_config = cfg_parser_.get_net_config();
    
    // Reset statistics
    layer_stats_.clear();
    layer_stats_.resize(layers.size());
    
    if (verbose_) {
        std::cout << "\nRunning forward pass on " << calibration_images.size() 
                  << " calibration image(s)..." << std::endl;
    }
    
    // Process each calibration image
    for (size_t img_idx = 0; img_idx < calibration_images.size(); img_idx++) {
        if (verbose_) {
            std::cout << "Processing image " << (img_idx + 1) << "/" 
                      << calibration_images.size() << ": " 
                      << calibration_images[img_idx] << std::endl;
        }
        
        // Load and preprocess image
        std::vector<float> input = load_and_preprocess_image(
            calibration_images[img_idx],
            net_config.width,
            net_config.height
        );
        
        if (input.empty()) {
            std::cerr << "Warning: Skipping image " << calibration_images[img_idx] << std::endl;
            continue;
        }
        
        // Open weights file for this forward pass
        WeightsReader weights_reader(verbose_);
        if (!weights_reader.open(weights_path)) {
            std::cerr << "Error: Failed to open weights file: " << weights_path << std::endl;
            return std::vector<int32_t>();
        }
        weights_reader.read_header();
        
        // Run forward pass
        forward_pass(input, weights_reader, bn_folder);
        
        weights_reader.close();
    }
    
    // Compute Q values from aggregated statistics
    std::vector<int32_t> q_values;
    q_values.push_back(14);  // First layer input Q14
    
    // Find Q value for each convolutional layer's output
    int conv_idx = 0;
    for (size_t i = 0; i < layers.size(); i++) {
        if (layers[i].is_convolutional()) {
            const ActivationStats& stats = layer_stats_[i];
            if (stats.initialized) {
                int q = quantizer.find_max_q(stats.min_val, stats.max_val);
                q_values.push_back(q);
                
                if (verbose_) {
                    std::cout << "Layer " << (conv_idx + 1) << " output: "
                              << "min=" << stats.min_val << ", max=" << stats.max_val
                              << ", Q=" << q << std::endl;
                }
            } else {
                // Fallback: use weight Q if no activation stats
                std::cerr << "Warning: No activation stats for layer " << i 
                          << ", using default Q" << std::endl;
                q_values.push_back(0);
            }
            conv_idx++;
        }
    }
    
    return q_values;
}

void ActivationTracker::forward_pass(
    const std::vector<float>& input,
    WeightsReader& weights_reader,
    BatchNormFolder& bn_folder
) {
    auto layers = cfg_parser_.get_layers();
    const NetworkConfig& net_config = cfg_parser_.get_net_config();
    
    // Track outputs for each layer (needed for route/shortcut)
    std::map<int, std::vector<float>> layer_outputs;
    std::map<int, std::tuple<int, int, int>> layer_dims;  // (w, h, c) for each layer
    
    int current_w = net_config.width;
    int current_h = net_config.height;
    int current_c = net_config.channels;
    std::vector<float> current_data = input;
    
    layer_dims[-1] = std::make_tuple(current_w, current_h, current_c);  // Input layer
    
    for (size_t i = 0; i < layers.size(); i++) {
        const LayerConfig& layer = layers[i];
        
        if (layer.is_convolutional()) {
            int n = layer.filters;
            // Use actual current channel count, not layer.channels (which may be outdated)
            int c = current_c;
            int groups = (layer.groups > 0) ? layer.groups : 1;
            int num_weights = (c / groups) * n * layer.size * layer.size;
            
            // Verify input size matches expected dimensions
            int expected_input_size = current_w * current_h * current_c;
            if (static_cast<int>(current_data.size()) != expected_input_size) {
                if (verbose_) {
                    std::cerr << "Error: Layer " << i << " (conv) input size mismatch. Expected " 
                              << expected_input_size << " (W=" << current_w << " H=" << current_h 
                              << " C=" << current_c << "), got " << current_data.size() << std::endl;
                    std::cerr << "  Config says layer.channels=" << layer.channels 
                              << ", but current_c=" << current_c << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            // Read weights and biases
            auto biases = weights_reader.read_biases(n);
            std::vector<float> scales, means, variances;
            if (layer.batch_normalize) {
                scales = weights_reader.read_scales(n);
                means = weights_reader.read_mean(n);
                variances = weights_reader.read_variance(n);
            }
            auto weights = weights_reader.read_weights(num_weights);
            
            // Fold batch normalization
            FoldedWeights folded = bn_folder.fold(
                weights, biases, scales, means, variances,
                n, num_weights / n, layer.batch_normalize
            );
            
            // Store input dimensions before convolution
            int in_w = current_w;
            int in_h = current_h;
            int in_c = current_c;
            
            // Apply convolution with proper dimensions
            current_data = apply_convolution_with_dims(current_data, folded.weights, 
                                                      folded.biases, layer,
                                                      in_w, in_h, in_c);
            
            // Check if convolution succeeded
            if (current_data.empty()) {
                if (verbose_) {
                    std::cerr << "Error: Convolution failed for layer " << i << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            // Calculate output dimensions after convolution
            // Note: apply_convolution_with_dims already calculates output size internally
            // We need to recalculate here to update current_w and current_h
            int out_w, out_h;
            calc_conv_output_size(in_w, in_h, layer.size, layer.stride, layer.pad, out_w, out_h);
            current_w = out_w;
            current_h = out_h;
            current_c = n;
            
            // Verify output size matches expected dimensions
            int expected_output_size = current_w * current_h * current_c;
            if (static_cast<int>(current_data.size()) != expected_output_size) {
                if (verbose_) {
                    std::cerr << "Error: Layer " << i << " output size mismatch. Expected " 
                              << expected_output_size << " (W=" << current_w << " H=" << current_h 
                              << " C=" << current_c << "), got " << current_data.size() << std::endl;
                    std::cerr << "  Input was: W=" << in_w << " H=" << in_h << " C=" << in_c << std::endl;
                    std::cerr << "  Conv params: size=" << layer.size << " stride=" << layer.stride 
                              << " pad=" << layer.pad << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            // Apply activation
            apply_activation(current_data, layer.activation);
            
            // Track activation statistics (after activation)
            for (float val : current_data) {
                layer_stats_[i].update(val);
            }
            
            layer_outputs[static_cast<int>(i)] = current_data;
            layer_dims[static_cast<int>(i)] = std::make_tuple(current_w, current_h, current_c);
            
            if (verbose_) {
                std::cerr << "Layer " << i << " (conv): " << in_w << "x" << in_h << "x" << in_c 
                          << " -> " << current_w << "x" << current_h << "x" << current_c << std::endl;
            }
            
        } else if (layer.is_maxpool()) {
            // Verify input size before maxpool
            int expected_input_size = current_w * current_h * current_c;
            if (static_cast<int>(current_data.size()) != expected_input_size) {
                if (verbose_) {
                    std::cerr << "Error: Maxpool layer " << i << " input size mismatch. Expected " 
                              << expected_input_size << " (W=" << current_w << " H=" << current_h 
                              << " C=" << current_c << "), got " << current_data.size() << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            // Store input dimensions before maxpool
            int in_w = current_w;
            int in_h = current_h;
            int in_c = current_c;
            
            current_data = apply_maxpool(current_data, layer, in_w, in_h, in_c);
            
            // Update dimensions after maxpool
            // Maxpool output: floor((input - pool_size) / pool_stride) + 1
            // For non-overlapping maxpool (pool_size == pool_stride): output = input / pool_stride
            int out_w = (in_w - layer.pool_size) / layer.pool_stride + 1;
            int out_h = (in_h - layer.pool_size) / layer.pool_stride + 1;
            current_w = out_w;
            current_h = out_h;
            // Channels remain the same after maxpool
            
            // Verify output size matches expected dimensions
            int expected_output_size = current_w * current_h * current_c;
            if (static_cast<int>(current_data.size()) != expected_output_size) {
                if (verbose_) {
                    std::cerr << "Error: Maxpool layer " << i << " output size mismatch. Expected " 
                              << expected_output_size << " (W=" << current_w << " H=" << current_h 
                              << " C=" << current_c << "), got " << current_data.size() << std::endl;
                    std::cerr << "  Input was: W=" << in_w << " H=" << in_h << " C=" << in_c << std::endl;
                    std::cerr << "  Pool params: size=" << layer.pool_size << " stride=" << layer.pool_stride << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            layer_outputs[static_cast<int>(i)] = current_data;
            layer_dims[static_cast<int>(i)] = std::make_tuple(current_w, current_h, current_c);
            
            if (verbose_) {
                std::cerr << "Layer " << i << " (maxpool): " << in_w << "x" << in_h << "x" << in_c 
                          << " -> " << current_w << "x" << current_h << "x" << current_c << std::endl;
            }
            
        } else if (layer.is_route()) {
            // Route concatenates channels from referenced layers
            // All referenced layers must have the same spatial dimensions (W, H)
            std::vector<float> route_data;
            int total_channels = 0;
            int route_w = -1, route_h = -1;
            bool first_layer = true;
            
            for (int ref : layer.layers) {
                int idx = cfg_parser_.resolve_layer_index(static_cast<int>(i), ref);
                
            if (verbose_) {
                std::cerr << "  Route layer " << i << " ref=" << ref << " -> idx=" << idx << std::endl;
            }
            
            if (layer_outputs.find(idx) != layer_outputs.end() && 
                layer_dims.find(idx) != layer_dims.end()) {
                const std::vector<float>& ref_data = layer_outputs[idx];
                int ref_w = std::get<0>(layer_dims[idx]);
                int ref_h = std::get<1>(layer_dims[idx]);
                int ref_c = std::get<2>(layer_dims[idx]);
                
                if (verbose_) {
                    std::cerr << "    Layer " << idx << ": " << ref_w << "x" << ref_h 
                              << "x" << ref_c << " (size=" << ref_data.size() << ")" << std::endl;
                }
                    
                    // Verify reference data size matches expected dimensions
                    int expected_ref_size = ref_w * ref_h * ref_c;
                    if (static_cast<int>(ref_data.size()) != expected_ref_size) {
                        if (verbose_) {
                            std::cerr << "Error: Route layer " << i << " reference layer " << idx 
                                      << " size mismatch. Expected " << expected_ref_size 
                                      << " (W=" << ref_w << " H=" << ref_h << " C=" << ref_c 
                                      << "), got " << ref_data.size() << std::endl;
                        }
                        return;  // Skip this forward pass
                    }
                    
                    // Verify spatial dimensions match (or set them from first layer)
                    if (first_layer) {
                        route_w = ref_w;
                        route_h = ref_h;
                        first_layer = false;
                    } else {
                        if (ref_w != route_w || ref_h != route_h) {
                            if (verbose_) {
                                std::cerr << "Error: Route layer " << i 
                                          << " spatial dimension mismatch. Layer " << idx 
                                          << " has " << ref_w << "x" << ref_h 
                                          << ", expected " << route_w << "x" << route_h << std::endl;
                                std::cerr << "  First layer had: " << route_w << "x" << route_h << std::endl;
                            }
                            return;  // Skip this forward pass
                        }
                    }
                    
                    route_data.insert(route_data.end(), ref_data.begin(), ref_data.end());
                    total_channels += ref_c;
                } else {
                    if (verbose_) {
                        std::cerr << "Error: Route layer " << i << " cannot find layer " << idx 
                                  << " (ref=" << ref << ")" << std::endl;
                        if (layer_outputs.find(idx) == layer_outputs.end()) {
                            std::cerr << "  Layer output not found" << std::endl;
                        }
                        if (layer_dims.find(idx) == layer_dims.end()) {
                            std::cerr << "  Layer dimensions not found" << std::endl;
                        }
                    }
                    return;  // Skip this forward pass
                }
            }
            
            // Verify route output size
            int expected_route_size = route_w * route_h * total_channels;
            if (static_cast<int>(route_data.size()) != expected_route_size) {
                if (verbose_) {
                    std::cerr << "Error: Route layer " << i << " output size mismatch. Expected " 
                              << expected_route_size << " (W=" << route_w << " H=" << route_h 
                              << " C=" << total_channels << "), got " << route_data.size() << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            current_data = route_data;
            current_c = total_channels;
            current_w = route_w;
            current_h = route_h;
            layer_outputs[static_cast<int>(i)] = current_data;
            layer_dims[static_cast<int>(i)] = std::make_tuple(current_w, current_h, current_c);
            
            if (verbose_) {
                std::cerr << "  Route layer " << i << " output: " << current_w << "x" << current_h 
                          << "x" << current_c << std::endl;
            }
            
        } else if (layer.is_shortcut()) {
            int from_idx = layer.layers.empty() ? static_cast<int>(i) - 1
                                               : cfg_parser_.resolve_layer_index(static_cast<int>(i), layer.layers[0]);
            std::vector<float> from_data;
            if (layer_outputs.find(from_idx) != layer_outputs.end()) {
                from_data = layer_outputs[from_idx];
            } else {
                if (verbose_) {
                    std::cerr << "Error: Shortcut layer " << i << " cannot find layer " << from_idx << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            // Verify dimensions match for shortcut
            if (layer_dims.find(from_idx) != layer_dims.end()) {
                int from_w = std::get<0>(layer_dims[from_idx]);
                int from_h = std::get<1>(layer_dims[from_idx]);
                int from_c = std::get<2>(layer_dims[from_idx]);
                
                if (from_w != current_w || from_h != current_h || from_c != current_c) {
                    if (verbose_) {
                        std::cerr << "Error: Shortcut layer " << i << " dimension mismatch. Current: " 
                                  << current_w << "x" << current_h << "x" << current_c 
                                  << ", From layer " << from_idx << ": " << from_w << "x" << from_h << "x" << from_c << std::endl;
                    }
                    return;  // Skip this forward pass
                }
            }
            
            current_data = apply_shortcut(current_data, from_data);
            layer_outputs[static_cast<int>(i)] = current_data;
            layer_dims[static_cast<int>(i)] = std::make_tuple(current_w, current_h, current_c);
            
            if (verbose_) {
                std::cerr << "Layer " << i << " (shortcut): " << current_w << "x" << current_h 
                          << "x" << current_c << " (from layer " << from_idx << ")" << std::endl;
            }
            
        } else if (layer.is_reorg()) {
            // Verify input size before reorg
            int expected_input_size = current_w * current_h * current_c;
            if (static_cast<int>(current_data.size()) != expected_input_size) {
                if (verbose_) {
                    std::cerr << "Error: Reorg layer " << i << " input size mismatch. Expected " 
                              << expected_input_size << " (W=" << current_w << " H=" << current_h 
                              << " C=" << current_c << "), got " << current_data.size() << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            // Store input dimensions
            int in_w = current_w;
            int in_h = current_h;
            int in_c = current_c;
            
            current_data = apply_reorg(current_data, layer.stride, in_w, in_h, in_c);
            
            // Update dimensions after reorg
            // Reorg divides spatial dimensions by stride and multiplies channels by stride^2
            current_w = in_w / layer.stride;
            current_h = in_h / layer.stride;
            current_c = in_c * layer.stride * layer.stride;
            
            // Verify output size
            int expected_output_size = current_w * current_h * current_c;
            if (static_cast<int>(current_data.size()) != expected_output_size) {
                if (verbose_) {
                    std::cerr << "Error: Reorg layer " << i << " output size mismatch. Expected " 
                              << expected_output_size << " (W=" << current_w << " H=" << current_h 
                              << " C=" << current_c << "), got " << current_data.size() << std::endl;
                }
                return;  // Skip this forward pass
            }
            
            layer_outputs[static_cast<int>(i)] = current_data;
            layer_dims[static_cast<int>(i)] = std::make_tuple(current_w, current_h, current_c);
            
            if (verbose_) {
                std::cerr << "Layer " << i << " (reorg): " << in_w << "x" << in_h << "x" << in_c 
                          << " -> " << current_w << "x" << current_h << "x" << current_c << std::endl;
            }
        } else if (layer.is_upsample()) {
            // Upsample layer: multiplies spatial dimensions by stride
            int in_w = current_w;
            int in_h = current_h;
            int in_c = current_c;
            
            current_w = in_w * layer.stride;
            current_h = in_h * layer.stride;
            // Channels remain the same
            
            // For now, upsample just duplicates data (simple implementation)
            std::vector<float> upsampled_data(current_w * current_h * current_c);
            for (int c = 0; c < in_c; c++) {
                for (int y = 0; y < in_h; y++) {
                    for (int x = 0; x < in_w; x++) {
                        int in_idx = (y * in_w + x) * in_c + c;
                        float val = current_data[in_idx];
                        for (int sy = 0; sy < layer.stride; sy++) {
                            for (int sx = 0; sx < layer.stride; sx++) {
                                int out_y = y * layer.stride + sy;
                                int out_x = x * layer.stride + sx;
                                int out_idx = (out_y * current_w + out_x) * current_c + c;
                                if (out_idx < static_cast<int>(upsampled_data.size())) {
                                    upsampled_data[out_idx] = val;
                                }
                            }
                        }
                    }
                }
            }
            current_data = upsampled_data;
            
            layer_outputs[static_cast<int>(i)] = current_data;
            layer_dims[static_cast<int>(i)] = std::make_tuple(current_w, current_h, current_c);
            
            if (verbose_) {
                std::cerr << "Layer " << i << " (upsample): " << in_w << "x" << in_h << "x" << in_c 
                          << " -> " << current_w << "x" << current_h << "x" << current_c << std::endl;
            }
        }
        // Skip other layer types (yolo, region, etc.)
    }
}

std::vector<float> ActivationTracker::apply_convolution_with_dims(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const LayerConfig& layer,
    int in_w,
    int in_h,
    int in_c
) {
    // Verify input size matches dimensions
    int expected_size = in_w * in_h * in_c;
    if (static_cast<int>(input.size()) != expected_size) {
        std::cerr << "Error: Input size mismatch. Expected " << expected_size 
                  << ", got " << input.size() << std::endl;
        return std::vector<float>();
    }
    
    int out_w, out_h;
    calc_conv_output_size(in_w, in_h, layer.size, layer.stride, layer.pad, out_w, out_h);
    
    // Calculate actual padding: pad=1 means "same" padding = (kernel-1)/2
    int actual_pad = (layer.pad == 1) ? ((layer.size - 1) / 2) : 0;
    
    int out_c = layer.filters;
    int groups = (layer.groups > 0) ? layer.groups : 1;
    int in_c_per_group = in_c / groups;
    
    std::vector<float> output(out_w * out_h * out_c, 0.0f);
    
    // Simple convolution implementation
    for (int g = 0; g < groups; g++) {
        int filter_start = g * (out_c / groups);
        int filter_end = (g + 1) * (out_c / groups);
        int in_c_start = g * in_c_per_group;
        
        for (int f = filter_start; f < filter_end; f++) {
            for (int y = 0; y < out_h; y++) {
                for (int x = 0; x < out_w; x++) {
                    float sum = biases[f];
                    
                    for (int ky = 0; ky < layer.size; ky++) {
                        for (int kx = 0; kx < layer.size; kx++) {
                            int in_y = y * layer.stride + ky - actual_pad;
                            int in_x = x * layer.stride + kx - actual_pad;
                            
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                for (int c = 0; c < in_c_per_group; c++) {
                                    int weight_idx = (f - filter_start) * in_c_per_group * layer.size * layer.size +
                                                    c * layer.size * layer.size + ky * layer.size + kx;
                                    int input_idx = (in_y * in_w + in_x) * in_c + (in_c_start + c);
                                    
                                    if (weight_idx < static_cast<int>(weights.size()) && 
                                        input_idx < static_cast<int>(input.size())) {
                                        sum += weights[weight_idx] * input[input_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    int out_idx = (y * out_w + x) * out_c + f;
                    if (out_idx < static_cast<int>(output.size())) {
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }
    
    return output;
}

void ActivationTracker::apply_activation(std::vector<float>& data, const std::string& activation) {
    if (activation == "leaky") {
        for (float& val : data) {
            if (val < 0.0f) {
                val = val * 0.1f;  // Leaky ReLU with alpha=0.1
            }
        }
    } else if (activation == "relu") {
        for (float& val : data) {
            if (val < 0.0f) {
                val = 0.0f;
            }
        }
    }
    // "linear" activation does nothing
}

std::vector<float> ActivationTracker::apply_maxpool(
    const std::vector<float>& input,
    const LayerConfig& layer,
    int input_w,
    int input_h,
    int input_c
) {
    int out_w = (input_w - layer.pool_size) / layer.pool_stride + 1;
    int out_h = (input_h - layer.pool_size) / layer.pool_stride + 1;
    
    std::vector<float> output(out_w * out_h * input_c);
    
    for (int c = 0; c < input_c; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float max_val = std::numeric_limits<float>::lowest();
                
                for (int py = 0; py < layer.pool_size; py++) {
                    for (int px = 0; px < layer.pool_size; px++) {
                        int in_y = y * layer.pool_stride + py;
                        int in_x = x * layer.pool_stride + px;
                        
                        if (in_y < input_h && in_x < input_w) {
                            int idx = (in_y * input_w + in_x) * input_c + c;
                            if (idx < static_cast<int>(input.size()) && input[idx] > max_val) {
                                max_val = input[idx];
                            }
                        }
                    }
                }
                
                int out_idx = (y * out_w + x) * input_c + c;
                if (out_idx < static_cast<int>(output.size())) {
                    output[out_idx] = max_val;
                }
            }
        }
    }
    
    return output;
}

// apply_route is now handled inline in forward_pass for better dimension tracking

std::vector<float> ActivationTracker::apply_shortcut(
    const std::vector<float>& current,
    const std::vector<float>& from
) {
    std::vector<float> result = current;
    
    if (from.size() == current.size()) {
        for (size_t i = 0; i < result.size(); i++) {
            result[i] += from[i];
        }
    }
    
    return result;
}

std::vector<float> ActivationTracker::apply_reorg(
    const std::vector<float>& input,
    int stride,
    int input_w,
    int input_h,
    int input_c
) {
    int out_w = input_w / stride;
    int out_h = input_h / stride;
    int out_c = input_c * stride * stride;
    
    std::vector<float> output(out_w * out_h * out_c, 0.0f);
    
    // Reorg reorganizes spatial dimensions into channels
    // For each stride x stride block in input, reorganize into channels in output
    for (int out_y = 0; out_y < out_h; out_y++) {
        for (int out_x = 0; out_x < out_w; out_x++) {
            for (int out_c_idx = 0; out_c_idx < out_c; out_c_idx++) {
                // Calculate which input position this output channel comes from
                int c = out_c_idx / (stride * stride);
                int pos_in_block = out_c_idx % (stride * stride);
                int block_x = pos_in_block % stride;
                int block_y = pos_in_block / stride;
                
                int in_x = out_x * stride + block_x;
                int in_y = out_y * stride + block_y;
                
                if (in_x < input_w && in_y < input_h && c < input_c) {
                    int in_idx = (in_y * input_w + in_x) * input_c + c;
                    int out_idx = (out_y * out_w + out_x) * out_c + out_c_idx;
                    
                    if (in_idx < static_cast<int>(input.size()) && 
                        out_idx < static_cast<int>(output.size())) {
                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
    }
    
    return output;
}

void ActivationTracker::calc_conv_output_size(int input_w, int input_h, int kernel, int stride, int pad,
                                               int& output_w, int& output_h) const {
    // In Darknet, pad=1 means "same" padding, which is calculated as (kernel-1)/2
    // pad=0 means no padding
    int actual_pad = (pad == 1) ? ((kernel - 1) / 2) : 0;
    
    output_w = (input_w + 2 * actual_pad - kernel) / stride + 1;
    output_h = (input_h + 2 * actual_pad - kernel) / stride + 1;
}

} // namespace darknet


/**
 * Activation tracker for computing IOFM Q values from calibration images
 */

#ifndef ACTIVATION_TRACKER_H
#define ACTIVATION_TRACKER_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <algorithm>
#include <cmath>
#include "cfg_parser.h"
#include "batch_norm_folder.h"
#include "int16_quantizer.h"
#include "weights_reader.h"

namespace darknet {

/**
 * Activation statistics for a layer
 */
struct ActivationStats {
    float min_val;
    float max_val;
    bool initialized;
    
    ActivationStats() : min_val(0.0f), max_val(0.0f), initialized(false) {}
    
    void update(float val) {
        if (!initialized) {
            min_val = max_val = val;
            initialized = true;
        } else {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    void merge(const ActivationStats& other) {
        if (!other.initialized) return;
        if (!initialized) {
            *this = other;
        } else {
            if (other.min_val < min_val) min_val = other.min_val;
            if (other.max_val > max_val) max_val = other.max_val;
        }
    }
};

/**
 * Activation tracker for forward pass
 */
class ActivationTracker {
public:
    ActivationTracker(const CfgParser& cfg_parser, bool verbose = false);
    ~ActivationTracker();
    
    /**
     * Process calibration images and compute activation Q values
     * 
     * @param calibration_images List of image file paths
     * @param weights_path Path to weights file (needed to reopen for each image)
     * @param bn_folder Batch normalization folder
     * @param quantizer INT16 quantizer for finding Q values
     * @return Vector of Q values for each layer's output (activation-based)
     */
    std::vector<int32_t> compute_activation_q_values(
        const std::vector<std::string>& calibration_images,
        const std::string& weights_path,
        BatchNormFolder& bn_folder,
        Int16Quantizer& quantizer
    );
    
    /**
     * Load and preprocess image (letterbox + normalization)
     * 
     * @param image_path Path to image file
     * @param target_w Target width
     * @param target_h Target height
     * @return Preprocessed image data (HWC format, normalized to [0,1])
     */
    std::vector<float> load_and_preprocess_image(
        const std::string& image_path,
        int target_w,
        int target_h
    );

private:
    const CfgParser& cfg_parser_;
    bool verbose_;
    std::vector<ActivationStats> layer_stats_;
    
    /**
     * Forward pass through the network
     * 
     * @param input Preprocessed input image
     * @param weights_reader Weights reader
     * @param bn_folder Batch normalization folder
     */
    void forward_pass(
        const std::vector<float>& input,
        WeightsReader& weights_reader,
        BatchNormFolder& bn_folder
    );
    
    /**
     * Apply convolution operation
     * 
     * @param input Input feature map
     * @param weights Convolution weights
     * @param biases Convolution biases
     * @param layer Layer configuration
     * @param in_w Input width
     * @param in_h Input height
     * @param in_c Input channels
     * @return Output feature map
     */
    std::vector<float> apply_convolution_with_dims(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const LayerConfig& layer,
        int in_w,
        int in_h,
        int in_c
    );
    
    /**
     * Apply activation function
     * 
     * @param data Feature map data
     * @param activation Activation type ("leaky", "linear", etc.)
     */
    void apply_activation(std::vector<float>& data, const std::string& activation);
    
    /**
     * Apply maxpool operation
     * 
     * @param input Input feature map
     * @param layer Layer configuration
     * @return Output feature map
     */
    std::vector<float> apply_maxpool(
        const std::vector<float>& input,
        const LayerConfig& layer,
        int input_w,
        int input_h,
        int input_c
    );
    
    // apply_route is handled inline in forward_pass for better dimension tracking
    
    /**
     * Apply shortcut (residual) operation
     * 
     * @param current Current feature map
     * @param from Feature map to add
     * @return Output feature map
     */
    std::vector<float> apply_shortcut(
        const std::vector<float>& current,
        const std::vector<float>& from
    );
    
    /**
     * Apply reorg operation
     * 
     * @param input Input feature map
     * @param stride Reorg stride
     * @param input_w Input width
     * @param input_h Input height
     * @param input_c Input channels
     * @return Reorganized output feature map
     */
    std::vector<float> apply_reorg(
        const std::vector<float>& input,
        int stride,
        int input_w,
        int input_h,
        int input_c
    );
    
    /**
     * Calculate output dimensions for convolution
     */
    void calc_conv_output_size(int input_w, int input_h, int kernel, int stride, int pad,
                               int& output_w, int& output_h) const;
};

} // namespace darknet

#endif // ACTIVATION_TRACKER_H


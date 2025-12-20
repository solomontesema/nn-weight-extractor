/**
 * Darknet weights to binary format extractor
 * 
 * Extracts weights and biases from Darknet .weights files and converts them
 * to binary format with batch normalization folding for FPGA acceleration.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include "cfg_parser.h"
#include "weights_reader.h"
#include "batch_norm_folder.h"
#include "int16_quantizer.h"
#include "activation_tracker.h"
#include <dirent.h>
#include <algorithm>

using namespace darknet;

// Command-line arguments structure
struct Arguments {
    std::string cfg_path;
    std::string weights_path;
    std::string output_dir = "outputs";
    std::string output_weights = "outputs/weights.bin";
    std::string output_bias = "outputs/bias.bin";
    bool int16_quantize = false;
    std::string output_weights_int16 = "outputs/weight_int16.bin";
    std::string output_bias_int16 = "outputs/bias_int16.bin";
    std::string output_weights_int16_q = "outputs/weight_int16_Q.bin";
    std::string output_bias_int16_q = "outputs/bias_int16_Q.bin";
    std::string output_iofm_q = "outputs/iofm_Q.bin";
    std::string calibration_folder;  // Folder containing calibration images
    bool verbose = false;
    bool help = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Extract weights and biases from Darknet format to binary files\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  --cfg PATH              Path to Darknet .cfg file\n";
    std::cout << "  --weights PATH          Path to Darknet .weights file\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  --output-dir PATH       Output directory (default: outputs)\n";
    std::cout << "  --output-weights PATH   Output weights file (default: outputs/weights.bin)\n";
    std::cout << "  --output-bias PATH      Output bias file (default: outputs/bias.bin)\n";
    std::cout << "  --int16                 Enable INT16 quantization\n";
    std::cout << "  --output-weights-int16 PATH   Output INT16 weights file (default: outputs/weight_int16.bin)\n";
    std::cout << "  --output-bias-int16 PATH      Output INT16 bias file (default: outputs/bias_int16.bin)\n";
    std::cout << "  --output-weights-int16-q PATH Output INT16 Q values file (default: outputs/weight_int16_Q.bin)\n";
    std::cout << "  --output-bias-int16-q PATH    Output INT16 Q values file (default: outputs/bias_int16_Q.bin)\n";
    std::cout << "  --output-iofm-q PATH          Output IOFM Q values file (default: outputs/iofm_Q.bin)\n";
    std::cout << "  --calib FOLDER                Folder containing calibration images (max 10 images used)\n";
    std::cout << "  --verbose, -v                 Enable verbose output\n";
    std::cout << "  --help, -h              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --cfg yolov2.cfg --weights yolov2.weights\n";
    std::cout << "  " << program_name << " --cfg model.cfg --weights model.weights --verbose\n";
    std::cout << "  " << program_name << " --cfg model.cfg --weights model.weights \\\n";
    std::cout << "      --output-weights out_w.bin --output-bias out_b.bin\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--cfg" && i + 1 < argc) {
            args.cfg_path = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            args.weights_path = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
            // Update default paths if output-dir is specified
            if (args.output_weights == "outputs/weights.bin") {
                args.output_weights = args.output_dir + "/weights.bin";
            }
            if (args.output_bias == "outputs/bias.bin") {
                args.output_bias = args.output_dir + "/bias.bin";
            }
            if (args.output_weights_int16 == "outputs/weight_int16.bin") {
                args.output_weights_int16 = args.output_dir + "/weight_int16.bin";
            }
            if (args.output_bias_int16 == "outputs/bias_int16.bin") {
                args.output_bias_int16 = args.output_dir + "/bias_int16.bin";
            }
            if (args.output_weights_int16_q == "outputs/weight_int16_Q.bin") {
                args.output_weights_int16_q = args.output_dir + "/weight_int16_Q.bin";
            }
            if (args.output_bias_int16_q == "outputs/bias_int16_Q.bin") {
                args.output_bias_int16_q = args.output_dir + "/bias_int16_Q.bin";
            }
        } else if (arg == "--output-weights" && i + 1 < argc) {
            args.output_weights = argv[++i];
        } else if (arg == "--output-bias" && i + 1 < argc) {
            args.output_bias = argv[++i];
        } else if (arg == "--int16") {
            args.int16_quantize = true;
        } else if (arg == "--output-weights-int16" && i + 1 < argc) {
            args.output_weights_int16 = argv[++i];
        } else if (arg == "--output-bias-int16" && i + 1 < argc) {
            args.output_bias_int16 = argv[++i];
        } else if (arg == "--output-weights-int16-q" && i + 1 < argc) {
            args.output_weights_int16_q = argv[++i];
        } else if (arg == "--output-bias-int16-q" && i + 1 < argc) {
            args.output_bias_int16_q = argv[++i];
        } else if (arg == "--output-iofm-q" && i + 1 < argc) {
            args.output_iofm_q = argv[++i];
        } else if (arg == "--calib" && i + 1 < argc) {
            args.calibration_folder = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
        }
    }
    
    return args;
}

bool validate_arguments(const Arguments& args) {
    if (args.help) {
        return false;
    }
    
    if (args.cfg_path.empty()) {
        std::cerr << "Error: --cfg argument is required\n";
        return false;
    }
    
    if (args.weights_path.empty()) {
        std::cerr << "Error: --weights argument is required\n";
        return false;
    }
    
    return true;
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    double value = static_cast<double>(bytes);
    int unit_index = 0;
    
    while (value >= 1024.0 && unit_index < 3) {
        value /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(value >= 10.0 ? 1 : 2)
        << value << " " << units[unit_index];
    return oss.str();
}

bool create_output_directory(const std::string& file_path) {
    // Extract directory from file path
    size_t last_slash = file_path.find_last_of("/\\");
    if (last_slash == std::string::npos) {
        return true;  // No directory component, file is in current directory
    }
    
    std::string dir = file_path.substr(0, last_slash);
    
    // Check if directory already exists
    struct stat info;
    if (stat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        return true;  // Directory exists
    }
    
    // Create directory (with parent directories if needed)
    // For simplicity, we'll create just the immediate directory
    // Using mode 0755 (rwxr-xr-x)
    if (mkdir(dir.c_str(), 0755) != 0) {
        // Check if it was created by another process
        if (stat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
            return true;
        }
        return false;
    }
    
    return true;
}

/**
 * Get image files from a directory (max 10 for calibration)
 */
std::vector<std::string> get_calibration_images(const std::string& folder_path, int max_images = 10) {
    std::vector<std::string> image_files;
    
    DIR* dir = opendir(folder_path.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open calibration folder: " << folder_path << std::endl;
        return image_files;
    }
    
    struct dirent* entry;
    std::vector<std::string> all_images;
    
    // Collect all image files
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename == "." || filename == "..") {
            continue;
        }
        
        // Check for common image extensions
        std::string lower_filename = filename;
        std::transform(lower_filename.begin(), lower_filename.end(), lower_filename.begin(), ::tolower);
        
        if (lower_filename.size() >= 4) {
            std::string ext = lower_filename.substr(lower_filename.size() - 4);
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp" || 
                (lower_filename.size() >= 5 && lower_filename.substr(lower_filename.size() - 5) == ".jpeg")) {
                all_images.push_back(folder_path + "/" + filename);
            }
        }
    }
    closedir(dir);
    
    // Sort for consistent selection
    std::sort(all_images.begin(), all_images.end());
    
    // Limit to max_images
    int count = std::min(static_cast<int>(all_images.size()), max_images);
    image_files.assign(all_images.begin(), all_images.begin() + count);
    
    return image_files;
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    Arguments args = parse_arguments(argc, argv);
    
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (!validate_arguments(args)) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "Darknet Weights Extractor" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;
    
    // Parse configuration file
    std::cout << "Parsing configuration file: " << args.cfg_path << std::endl;
    CfgParser cfg_parser(args.verbose);
    
    if (!cfg_parser.parse(args.cfg_path)) {
        std::cerr << "Error: Failed to parse config file" << std::endl;
        return 1;
    }
    
    auto conv_layers = cfg_parser.get_conv_layers();
    std::cout << "Found " << conv_layers.size() << " convolutional layers" << std::endl;
    
    if (args.verbose) {
        cfg_parser.print_summary();
    }
    
    // Open weights file
    std::cout << "\nReading weights file: " << args.weights_path << std::endl;
    WeightsReader weights_reader(args.verbose);
    
    if (!weights_reader.open(args.weights_path)) {
        std::cerr << "Error: Failed to open weights file" << std::endl;
        return 1;
    }
    
    // Read header (will be read again after activation computation if needed)
    weights_reader.read_header();
    
    // Create output directory if needed
    if (!create_output_directory(args.output_weights)) {
        std::cerr << "Error: Cannot create output directory for: " << args.output_weights << std::endl;
        return 1;
    }
    
    // Open output files
    std::ofstream weights_out(args.output_weights, std::ios::binary);
    std::ofstream bias_out(args.output_bias, std::ios::binary);
    
    if (!weights_out.is_open()) {
        std::cerr << "Error: Cannot open output weights file: " << args.output_weights << std::endl;
        return 1;
    }
    
    if (!bias_out.is_open()) {
        std::cerr << "Error: Cannot open output bias file: " << args.output_bias << std::endl;
        return 1;
    }
    
    // Open INT16 output files if quantization is enabled
    std::ofstream weights_int16_out;
    std::ofstream bias_int16_out;
    
    if (args.int16_quantize) {
        weights_int16_out.open(args.output_weights_int16, std::ios::binary);
        bias_int16_out.open(args.output_bias_int16, std::ios::binary);
        
        if (!weights_int16_out.is_open()) {
            std::cerr << "Error: Cannot open INT16 weights file: " << args.output_weights_int16 << std::endl;
            return 1;
        }
        
        if (!bias_int16_out.is_open()) {
            std::cerr << "Error: Cannot open INT16 bias file: " << args.output_bias_int16 << std::endl;
            return 1;
        }
    }
    
    std::cout << "\nExtracting and processing layers..." << std::endl;
    if (args.int16_quantize) {
        std::cout << "INT16 quantization enabled" << std::endl;
    }
    
    // Create batch norm folder
    BatchNormFolder bn_folder;
    
    // Create INT16 quantizer if needed
    Int16Quantizer* quantizer = nullptr;
    if (args.int16_quantize) {
        quantizer = new Int16Quantizer(args.verbose);
    }
    
    // Store Q values for all layers
    std::vector<int32_t> weight_q_values;
    std::vector<int32_t> bias_q_values;
    std::vector<int32_t> iofm_q_values;  // Input/Output Feature Map Q values
    
    // Compute activation-based IOFM Q values if calibration folder is provided
    // This must be done BEFORE processing layers so we have the Q values ready
    bool use_activation_q = false;
    if (args.int16_quantize && !args.calibration_folder.empty()) {
        if (args.verbose) {
            std::cout << "\nScanning calibration folder: " << args.calibration_folder << std::endl;
        }
        
        // Get calibration images from folder (max 10)
        std::vector<std::string> calibration_images = get_calibration_images(args.calibration_folder, 10);
        
        if (calibration_images.empty()) {
            std::cerr << "Error: No image files found in calibration folder: " 
                      << args.calibration_folder << std::endl;
            std::cerr << "Supported formats: .jpg, .jpeg, .png, .bmp" << std::endl;
            return 1;
        }
        
        if (args.verbose) {
            std::cout << "Found " << calibration_images.size() << " calibration image(s)" << std::endl;
            std::cout << "Computing activation-based IOFM Q values..." << std::endl;
        }
        
        // Create a temporary weights reader for forward pass
        WeightsReader temp_weights_reader(args.verbose);
        if (!temp_weights_reader.open(args.weights_path)) {
            std::cerr << "Error: Failed to open weights file for activation computation" << std::endl;
            return 1;
        }
        temp_weights_reader.read_header();
        temp_weights_reader.close();
        
        ActivationTracker activation_tracker(cfg_parser, args.verbose);
        iofm_q_values = activation_tracker.compute_activation_q_values(
            calibration_images,
            args.weights_path,
            bn_folder,
            *quantizer
        );
        
        if (iofm_q_values.empty()) {
            std::cerr << "\nWarning: Failed to compute activation Q values from forward pass." << std::endl;
            std::cerr << "This may be due to dimension tracking issues in complex network architectures." << std::endl;
            std::cerr << "Falling back to weight-based Q values for IOFM (may be less accurate)." << std::endl;
            std::cerr << "The tool will continue with weight extraction and quantization." << std::endl;
            use_activation_q = false;
            iofm_q_values.clear();
            iofm_q_values.push_back(14);  // First layer input Q14
        } else {
            use_activation_q = true;
        }
        
        if (args.verbose) {
            std::cout << "Computed " << iofm_q_values.size() << " activation-based IOFM Q values" << std::endl;
        }
    } else if (args.int16_quantize) {
        // Fallback: use weight Q values if no calibration folder
        if (args.verbose) {
            std::cout << "\nWarning: No calibration folder provided. " 
                      << "IOFM Q values will use weight Q values (may be inaccurate)." << std::endl;
            std::cout << "For accurate results, use --calib <folder_path>" << std::endl;
        }
        iofm_q_values.push_back(14);  // First layer input Q14
    }
    
    // Process each convolutional layer
    size_t total_weights = 0;
    size_t total_biases = 0;
    
    for (size_t i = 0; i < conv_layers.size(); i++) {
        const LayerConfig& layer = conv_layers[i];
        
        int n = layer.filters;
        int c = layer.channels;
        // Ensure groups is at least 1 to avoid division by zero
        int groups = (layer.groups > 0) ? layer.groups : 1;
        int num_weights = (c / groups) * n * layer.size * layer.size;
        
        if (args.verbose) {
            std::cout << "\nLayer " << (i+1) << "/" << conv_layers.size() 
                      << ": " << layer.type << std::endl;
            std::cout << "  Filters: " << layer.filters << std::endl;
            std::cout << "  Channels: " << layer.channels << std::endl;
            std::cout << "  Size: " << layer.size << "x" << layer.size << std::endl;
            std::cout << "  Stride: " << layer.stride << std::endl;
            std::cout << "  Groups: " << groups << std::endl;
            std::cout << "  Calculated num_weights: " << num_weights << std::endl;
            std::cout << "  Batch normalize: " << (layer.batch_normalize ? "yes" : "no") << std::endl;
        }
        
        // Read biases
        auto biases = weights_reader.read_biases(n);
        if (biases.size() != (size_t)n) {
            std::cerr << "Error: Failed to read biases for layer " << i << std::endl;
            return 1;
        }
        
        // Read batch normalization parameters if present
        std::vector<float> scales, means, variances;
        if (layer.batch_normalize) {
            scales = weights_reader.read_scales(n);
            means = weights_reader.read_mean(n);
            variances = weights_reader.read_variance(n);
            
            if (scales.size() != (size_t)n || means.size() != (size_t)n || 
                variances.size() != (size_t)n) {
                std::cerr << "Error: Failed to read BN parameters for layer " << i << std::endl;
                return 1;
            }
        }
        
        // Read weights
        auto weights = weights_reader.read_weights(num_weights);
        if (weights.size() != (size_t)num_weights) {
            std::cerr << "Error: Failed to read weights for layer " << i << std::endl;
            return 1;
        }
        
        // Fold batch normalization
        FoldedWeights folded = bn_folder.fold(
            weights, biases, scales, means, variances,
            n, num_weights / n, layer.batch_normalize
        );
        
        // Write to output files
        weights_out.write(reinterpret_cast<const char*>(folded.weights.data()), 
                         folded.weights.size() * sizeof(float));
        bias_out.write(reinterpret_cast<const char*>(folded.biases.data()),
                      folded.biases.size() * sizeof(float));
        
        // INT16 quantization if enabled
        if (args.int16_quantize && quantizer) {
            QuantizedData quantized = quantizer->quantize_layer(folded.weights, folded.biases);
            
            // Write quantized weights
            size_t weights_size = quantized.weights.size();
            weights_int16_out.write(reinterpret_cast<const char*>(quantized.weights.data()),
                                   weights_size * sizeof(int16_t));
            
            // Handle padding if odd number of elements
            if (weights_size & 0x1) {
                int16_t pad = 0;
                weights_int16_out.write(reinterpret_cast<const char*>(&pad), sizeof(int16_t));
            }
            
            // Write quantized biases
            size_t biases_size = quantized.biases.size();
            bias_int16_out.write(reinterpret_cast<const char*>(quantized.biases.data()),
                                biases_size * sizeof(int16_t));
            
            // Handle padding if odd number of elements
            if (biases_size & 0x1) {
                int16_t pad = 0;
                bias_int16_out.write(reinterpret_cast<const char*>(&pad), sizeof(int16_t));
            }
            
            // Store Q values (same for weights and biases in the same layer)
            weight_q_values.push_back(quantized.q_value);
            bias_q_values.push_back(quantized.q_value);
            
            // IOFM Q values: use activation-based if computed, otherwise fallback to weight Q
            if (!use_activation_q) {
                // Fallback: use weight Q for IOFM (less accurate)
                if (iofm_q_values.size() == 1) {
                    // First value is already Q14, now add output Q for this layer
                    iofm_q_values.push_back(quantized.q_value);
                } else {
                    iofm_q_values.push_back(quantized.q_value);
                }
            }
            // If use_activation_q is true, iofm_q_values already contains activation-based Q values
        }
        
        total_weights += folded.weights.size();
        total_biases += folded.biases.size();

        if (args.verbose) {
            // Detailed per-layer summary
            size_t weights_bytes = folded.weights.size() * sizeof(float);
            size_t bias_bytes = folded.biases.size() * sizeof(float);
            int in_channels = layer.channels;
            int out_channels = layer.filters;
            int kernel = layer.size;
            int groups = layer.groups;
            int in_per_group = (groups > 0) ? (in_channels / groups) : in_channels;

            std::cout << "Layer " << (i+1) << "/" << conv_layers.size()
                      << " [conv]"
                      << " out_ch=" << out_channels
                      << " in_ch=" << in_channels
                      << " groups=" << groups
                      << " kernel=" << kernel << "x" << kernel
                      << " (per-group in=" << in_per_group << ")"
                      << " weights=" << folded.weights.size() << " (" << format_bytes(weights_bytes) << ")"
                      << " bias=" << folded.biases.size() << " (" << format_bytes(bias_bytes) << ")"
                      << (layer.batch_normalize ? " BN-folded" : "");
            
            if (args.int16_quantize && !weight_q_values.empty()) {
                int q_val = weight_q_values.back();
                std::cout << " Q=" << q_val;
            }
            std::cout << std::endl;
        } else {
            // Compact progress for non-verbose runs
            std::cout << "\rProcessed layer " << (i+1) << "/" << conv_layers.size() << std::flush;
        }
    }

    if (!args.verbose) {
        std::cout << std::endl;
    }
    
    // Write Q values if quantization was enabled
    if (args.int16_quantize && !weight_q_values.empty()) {
        std::ofstream weights_q_out(args.output_weights_int16_q, std::ios::binary);
        std::ofstream bias_q_out(args.output_bias_int16_q, std::ios::binary);
        std::ofstream iofm_q_out(args.output_iofm_q, std::ios::binary);
        
        if (!weights_q_out.is_open()) {
            std::cerr << "Error: Cannot open Q values file: " << args.output_weights_int16_q << std::endl;
            return 1;
        }
        
        if (!bias_q_out.is_open()) {
            std::cerr << "Error: Cannot open Q values file: " << args.output_bias_int16_q << std::endl;
            return 1;
        }
        
        if (!iofm_q_out.is_open()) {
            std::cerr << "Error: Cannot open IOFM Q values file: " << args.output_iofm_q << std::endl;
            return 1;
        }
        
        // Write Q values as int32_t
        weights_q_out.write(reinterpret_cast<const char*>(weight_q_values.data()),
                           weight_q_values.size() * sizeof(int32_t));
        bias_q_out.write(reinterpret_cast<const char*>(bias_q_values.data()),
                        bias_q_values.size() * sizeof(int32_t));
        
        // Write IOFM Q values: [input_Q_layer0, output_Q_layer0, output_Q_layer1, ...]
        // First value is input Q for first layer (14), then output Q for each layer
        iofm_q_out.write(reinterpret_cast<const char*>(iofm_q_values.data()),
                        iofm_q_values.size() * sizeof(int32_t));
        
        weights_q_out.close();
        bias_q_out.close();
        iofm_q_out.close();
    }
    
    // Close files
    weights_reader.close();
    weights_out.close();
    bias_out.close();
    
    if (args.int16_quantize) {
        weights_int16_out.close();
        bias_int16_out.close();
    }
    
    // Clean up quantizer
    if (quantizer) {
        delete quantizer;
    }
    
    // Print summary
    std::cout << "\n=========================" << std::endl;
    std::cout << "Extraction completed successfully!" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Layers processed: " << conv_layers.size() << std::endl;
    std::cout << "  Total weights: " << total_weights << std::endl;
    std::cout << "  Total biases: " << total_biases << std::endl;
    std::cout << "  Total parameters: " << (total_weights + total_biases) << std::endl;
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  Weights: " << args.output_weights << std::endl;
    std::cout << "  Biases:  " << args.output_bias << std::endl;
    
    if (args.int16_quantize) {
        std::cout << "\nINT16 quantization files:" << std::endl;
        std::cout << "  Weights (INT16): " << args.output_weights_int16 << std::endl;
        std::cout << "  Biases (INT16):  " << args.output_bias_int16 << std::endl;
        std::cout << "  Weights Q values: " << args.output_weights_int16_q << std::endl;
        std::cout << "  Biases Q values:  " << args.output_bias_int16_q << std::endl;
        std::cout << "  IOFM Q values:    " << args.output_iofm_q << std::endl;
    }
    
    return 0;
}

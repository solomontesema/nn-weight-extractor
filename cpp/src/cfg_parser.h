/**
 * Darknet configuration file parser
 */

#ifndef CFG_PARSER_H
#define CFG_PARSER_H

#include <string>
#include <vector>
#include <map>

namespace darknet {

/**
 * Layer configuration structure
 */
struct LayerConfig {
    std::string type;         // Layer type: convolutional, maxpool, route, etc.
    int index;                // Layer index
    
    // Convolutional layer parameters
    int filters;              // Number of output filters
    int channels;             // Number of input channels
    int size;                 // Kernel size
    int stride;               // Stride
    int pad;                  // Padding (0 or 1)
    int groups;               // Group convolutions
    bool batch_normalize;     // Has batch normalization
    std::string activation;   // Activation function
    
    // Maxpool parameters
    int pool_size;
    int pool_stride;
    
    // Route/Shortcut parameters
    std::vector<int> layers;  // Referenced layer indices
    
    // Constructor with defaults
    LayerConfig() : 
        type(""), index(0), filters(0), channels(0), size(0), 
        stride(1), pad(0), groups(1), batch_normalize(false),
        activation("linear"), pool_size(0), pool_stride(0) {}
    
    bool is_convolutional() const { return type == "convolutional"; }
    bool is_maxpool() const { return type == "maxpool"; }
    bool is_route() const { return type == "route"; }
    bool is_shortcut() const { return type == "shortcut"; }
    bool is_upsample() const { return type == "upsample"; }
    bool is_reorg() const { return type == "reorg"; }
    bool is_yolo() const { return type == "yolo"; }
};

/**
 * Network configuration
 */
struct NetworkConfig {
    int width;
    int height;
    int channels;
    int batch;
    int subdivisions;
    float momentum;
    float decay;
    
    NetworkConfig() : 
        width(416), height(416), channels(3), 
        batch(1), subdivisions(1), momentum(0.9), decay(0.0005) {}
};

/**
 * Configuration file parser
 */
class CfgParser {
public:
    CfgParser(bool verbose = false);
    ~CfgParser();
    
    /**
     * Parse configuration file
     * @param cfg_path Path to .cfg file
     * @return true if successful
     */
    bool parse(const std::string& cfg_path);
    
    /**
     * Get network configuration
     */
    const NetworkConfig& get_network_config() const { return net_config_; }
    
    /**
     * Get all layer configurations
     */
    const std::vector<LayerConfig>& get_layers() const { return layers_; }
    
    /**
     * Get only convolutional layers
     */
    std::vector<LayerConfig> get_conv_layers() const;
    
    /**
     * Get layer at index
     */
    const LayerConfig* get_layer(int index) const;
    
    /**
     * Get number of layers
     */
    size_t num_layers() const { return layers_.size(); }
    
    /**
     * Print configuration summary
     */
    void print_summary() const;
    
    /**
     * Get network configuration (alias for get_network_config)
     */
    const NetworkConfig& get_net_config() const { return net_config_; }
    
    /**
     * Resolve layer index from reference (for route/shortcut layers)
     */
    int resolve_layer_index(int current_index, int reference) const;

private:
    bool verbose_;
    NetworkConfig net_config_;
    std::vector<LayerConfig> layers_;
    
    void compute_layer_channels();
    
    // Helper methods
    void parse_net_section(const std::map<std::string, std::string>& options);
    LayerConfig parse_convolutional(const std::map<std::string, std::string>& options, int index);
    LayerConfig parse_maxpool(const std::map<std::string, std::string>& options, int index);
    LayerConfig parse_route(const std::map<std::string, std::string>& options, int index);
    LayerConfig parse_shortcut(const std::map<std::string, std::string>& options, int index);
    LayerConfig parse_upsample(const std::map<std::string, std::string>& options, int index);
    LayerConfig parse_reorg(const std::map<std::string, std::string>& options, int index);
    LayerConfig parse_yolo(const std::map<std::string, std::string>& options, int index);
    
    int get_int_option(const std::map<std::string, std::string>& options, 
                      const std::string& key, int default_value) const;
    float get_float_option(const std::map<std::string, std::string>& options,
                          const std::string& key, float default_value) const;
    std::string get_string_option(const std::map<std::string, std::string>& options,
                                 const std::string& key, const std::string& default_value) const;
    bool has_option(const std::map<std::string, std::string>& options,
                   const std::string& key) const;
    
    std::string trim(const std::string& str) const;
    std::vector<int> parse_int_list(const std::string& str) const;
};

} // namespace darknet

#endif // CFG_PARSER_H

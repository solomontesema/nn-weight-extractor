# Neural Network Weight Extractor

A standalone tool for extracting and converting neural network weights from Keras/TensorFlow models to binary format with batch normalization folding and optional INT16 quantization. Designed for hardware acceleration workflows, embedded systems, and custom inference engines.

## Overview

This tool provides a clean two-step pipeline for converting trained neural network models into optimized binary weight and bias files:

```
Keras/TensorFlow Model → Darknet Format → Binary Files (weights.bin + bias.bin)
                                              ↓ (optional)
                                    INT16 Quantized Files (weight_int16.bin + bias_int16.bin)
```

The tool performs batch normalization folding during extraction, eliminating BN operations and reducing computational requirements for inference. Optional INT16 quantization with Q-format fixed-point representation further reduces memory footprint and bandwidth - ideal for FPGA, ASIC, and embedded deployments.

## Key Features

- **Multiple Input Formats**: Keras H5 models, Darknet .weights files
- **Batch Normalization Folding**: Automatically folds BN into convolutional layers
- **INT16 Quantization**: Optional Q-format fixed-point quantization for FPGA/ASIC deployment
- **Fast C++ Extractor**: Zero external dependencies, processes 50M parameters in <1 second
- **Architecture Support**: YOLOv2, YOLOv3, custom CNN architectures
- **Production Ready**: Comprehensive error handling, validation, and logging
- **Hardware Optimized**: Binary output format ready for hardware accelerators

## Installation

### Python Converter (Optional - for H5 models)

```bash
cd python
pip install -r requirements.txt
```

Requirements: Python 3.6+, TensorFlow/Keras, NumPy

### C++ Extractor

```bash
cd cpp
make
```

Requirements: C++11 compiler (g++, clang++)

## Quick Start

### Extract from Darknet Weights

If you already have Darknet .weights and .cfg files:

```bash
cd cpp
./weights_extractor --cfg model.cfg --weights model.weights
```

Output: `outputs/weights.bin` and `outputs/bias.bin` in the `outputs/` directory (created automatically if it doesn't exist).

**With INT16 quantization:**
```bash
./weights_extractor --cfg model.cfg --weights model.weights --int16
```

**With calibration-based quantization:**
```bash
./weights_extractor --cfg model.cfg --weights model.weights --int16 --calib /path/to/images
```

Output: Float32 files plus INT16 quantized files (`weight_int16.bin`, `bias_int16.bin`) and Q value files in the `outputs/` directory. When using `--calib`, IOFM Q values are derived from activation statistics for improved quantization accuracy.

### Convert from Keras H5

For complete Keras models (saved with `model.save()`):

```bash
# Step 1: Convert to Darknet format
cd python
python h5_to_darknet.py \
    --input model.h5 \
    --output-weights model.weights \
    --output-cfg model.cfg

# Step 2: Extract to binary
cd ../cpp
./weights_extractor --cfg model.cfg --weights model.weights
```

## Usage

### Python H5 Converter

```bash
python h5_to_darknet.py \
    --input model.h5 \
    --output-weights output.weights \
    --output-cfg output.cfg \
    [--img-size 416] \
    [--channels 3] \
    [--verbose]
```

**Options:**
- `--input`: Input Keras H5 model file
- `--output-weights`: Output Darknet weights file
- `--output-cfg`: Output Darknet configuration file
- `--img-size`: Input image size (default: from model)
- `--img-width/--img-height`: Specify width and height separately
- `--channels`: Number of input channels (default: 3)
- `--verbose`: Enable detailed output

**Note**: The H5 file must contain the complete model architecture (saved with `model.save()`, not `model.save_weights()`). For weights-only files, see the [YOLOv3 Guide](YOLOV3_GUIDE.md).

### C++ Binary Extractor

```bash
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    [--output-weights weights.bin] \
    [--output-bias bias.bin] \
    [--verbose]
```

**Options:**
- `--cfg`: Darknet configuration file
- `--weights`: Darknet weights file
- `--output-dir`: Output directory (default: outputs)
- `--output-weights`: Output weights binary (default: outputs/weights.bin)
- `--output-bias`: Output bias binary (default: outputs/bias.bin)
- `--int16`: Enable INT16 quantization (outputs INT16 files and Q values)
- `--calib <folder>`: Enable calibration-based quantization using images from specified folder (generates activation-based IOFM Q values)
- `--output-weights-int16`: Output INT16 weights file (default: outputs/weight_int16.bin)
- `--output-bias-int16`: Output INT16 bias file (default: outputs/bias_int16.bin)
- `--output-weights-int16-q`: Output Q values for weights (default: outputs/weight_int16_Q.bin)
- `--output-bias-int16-q`: Output Q values for biases (default: outputs/bias_int16_Q.bin)
- `--output-iofm-q`: Output IOFM Q values file (default: outputs/iofm_Q.bin)
- `--verbose`: Enable detailed layer-by-layer output

## What It Does

This tool extracts neural network weights and biases, then:

1. **Extracts weights** from Keras H5 or Darknet .weights formats
2. **Folds batch normalization** into convolutional layers for efficiency:
   ```
   new_weight = weight × (scale / √(variance + ε))
   new_bias = bias - mean × (scale / √(variance + ε))
   ```
3. **Converts to binary format** (32-bit floats, little-endian)
4. **Optionally quantizes to INT16** using Q-format fixed-point representation (when `--int16` flag is used):
   - Per-layer Q value selection (Q0-Q15) for optimal precision
   - Converts float32 weights and biases to int16_t format
   - Generates Q value files for dequantization during inference
   - Supports calibration-based quantization (with `--calib` flag) for activation-based IOFM Q values
   - Achieves 50% memory reduction while preserving accuracy
5. **Generates configuration** describing network architecture

### Why Batch Normalization Folding?

Folding BN into conv layers:
- ✓ Eliminates BN computations during inference
- ✓ Reduces memory bandwidth requirements  
- ✓ Maintains numerical accuracy
- ✓ Simplifies hardware implementation
- ✓ Reduces latency and power consumption

Perfect for FPGAs, ASICs, microcontrollers, and custom accelerators.

### Why INT16 Quantization?

When combined with INT16 quantization (via `--int16` flag):
- ✓ 50% memory reduction (2 bytes vs 4 bytes per value)
- ✓ Reduced memory bandwidth for hardware accelerators
- ✓ More efficient INT16 arithmetic on FPGAs
- ✓ Per-layer Q value optimization preserves precision
- ✓ Model-agnostic quantization works with any architecture

Together, BN folding and INT16 quantization provide a complete optimization pipeline for hardware deployment.

## Output Format

All output files are written to the `outputs/` directory by default (can be changed with `--output-dir`). The directory is created automatically if it doesn't exist.

### weights.bin
- Location: `outputs/weights.bin` (default)
- Binary file of 32-bit floats (IEEE 754, little-endian)
- Sequential convolutional layer weights (BN-folded)
- Format: `[layer0_weights, layer1_weights, ...]`

### bias.bin  
- Location: `outputs/bias.bin` (default)
- Binary file of 32-bit floats (IEEE 754, little-endian)
- Sequential convolutional layer biases (BN-folded)
- Format: `[layer0_biases, layer1_biases, ...]`

## INT16 Quantization

The tool supports optional INT16 quantization using Q-format fixed-point representation, ideal for FPGA and ASIC implementations where reduced precision and memory bandwidth are critical.

### How It Works

INT16 quantization converts float32 weights and biases to 16-bit integers using per-layer Q values:

- **Q-format**: Each layer gets a Q value (0-15) that determines the quantization scale
- **Quantization formula**: `int16_value = (int16_t)(float_value * 2^Q)`
- **Dequantization**: `float_value = (float)int16_value * 2^(-Q)`
- **Per-layer Q selection**: Automatically finds the highest Q value that can represent all values in a layer

### Calibration-Based Quantization

For optimal quantization accuracy, the tool supports calibration-based quantization using representative input images. This generates activation-based Q values for input/output feature maps (IOFM) by analyzing actual activation ranges during forward passes:

- **Activation-based Q values**: Q values are derived from actual activation statistics rather than weight statistics, providing better quantization accuracy
- **Calibration images**: Provide a folder containing representative images (`.jpg` format) from your dataset
- **Image selection**: If more than 10 images are provided, the tool uses the first 10 alphabetically ordered `.jpg` images for calibration
- **Input layer Q value**: The first input layer always uses a constant Q value of 14 (Q14)
- **Forward pass analysis**: The tool performs forward passes through the network to collect activation min/max statistics for each layer

**Q-format ranges:**
- Q0: [-32768, 32767] (largest range, lowest precision)
- Q15: [-1, 0.99996948] (smallest range, highest precision)

### Usage

Enable INT16 quantization with the `--int16` flag:

```bash
cd cpp
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    --int16
```

This generates five output files in the `outputs/` directory:
- `outputs/weight_int16.bin`: Quantized weights as int16_t values
- `outputs/bias_int16.bin`: Quantized biases as int16_t values  
- `outputs/weight_int16_Q.bin`: Q values for each layer's weights (int32_t array)
- `outputs/bias_int16_Q.bin`: Q values for each layer's biases (int32_t array)
- `outputs/iofm_Q.bin`: Q values for input/output feature maps (int32_t array)

**With calibration-based quantization:**

```bash
cd cpp
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    --int16 \
    --calib /path/to/calibration/images
```

When using `--calib`, the tool performs forward passes on calibration images to generate activation-based IOFM Q values. This provides more accurate quantization compared to weight-based Q values, especially for complex network architectures.

### Example: YOLOv2 INT16 Quantization

**Weight-based quantization:**
```bash
cd cpp
./weights_extractor \
    --cfg ../cfg/yolov2.cfg \
    --weights ../weights/yolov2.weights \
    --int16 \
    --verbose
```

**Calibration-based quantization:**
```bash
cd cpp
./weights_extractor \
    --cfg ../cfg/yolov2.cfg \
    --weights ../weights/yolov2.weights \
    --int16 \
    --calib ../test_images \
    --verbose
```

**Output:**
- `outputs/weights.bin` and `outputs/bias.bin` (float32, always generated)
- `outputs/weight_int16.bin` and `outputs/bias_int16.bin` (INT16 quantized)
- `outputs/weight_int16_Q.bin` and `outputs/bias_int16_Q.bin` (Q values per layer)
- `outputs/iofm_Q.bin` (Q values for input/output feature maps)
  - With `--calib`: Activation-based Q values derived from forward pass statistics
  - Without `--calib`: Weight-based Q values (fallback method)

### Custom Output Directory

```bash
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    --output-dir my_outputs
```

### Custom Output Paths

```bash
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    --int16 \
    --output-weights-int16 custom_weights_i16.bin \
    --output-bias-int16 custom_bias_i16.bin \
    --output-weights-int16-q custom_weights_q.bin \
    --output-bias-int16-q custom_bias_q.bin \
    --output-iofm-q custom_iofm_q.bin
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

### File Formats

**INT16 weight/bias files:**
- Binary format: `int16_t` values (2 bytes each, little-endian)
- Sequential layer-by-layer storage
- Padding: If a layer has odd number of elements, one zero-padded element is added

**Q value files:**
- Binary format: `int32_t` array (4 bytes per Q value, little-endian)
- `weight_int16_Q.bin`: One Q value per convolutional layer (0-15) for weights
- `bias_int16_Q.bin`: One Q value per convolutional layer (0-15) for biases
- `iofm_Q.bin`: Q values for input/output feature maps
  - First value: Constant Q14 (input Q for the first layer, always 14)
  - Subsequent values: Output Q for each convolutional layer (becomes input Q for next layer)
  - Total: `num_conv_layers + 1` values
  - When using `--calib`: Q values are derived from activation statistics during forward passes
  - Without `--calib`: Q values are derived from weight statistics (fallback method)

### Benefits

- **Memory reduction**: 50% size reduction (2 bytes vs 4 bytes per value)
- **Bandwidth savings**: Reduced memory bandwidth for hardware accelerators
- **FPGA-friendly**: INT16 arithmetic is more efficient on FPGAs
- **Maintains accuracy**: Per-layer Q selection preserves precision where possible
- **Model-agnostic**: Works with any Darknet cfg file
- **IOFM quantization**: Generates Q values for input/output feature maps for inference
- **Calibration support**: Activation-based quantization using representative images for improved accuracy

### Dequantization Example

To convert INT16 values back to float32 during inference:

```c
// Read Q values for layer i
int32_t weight_q = weight_q_values[i];
int32_t bias_q = bias_q_values[i];
int32_t input_q = iofm_q_values[i];      // Input Q for this layer
int32_t output_q = iofm_q_values[i+1];   // Output Q for this layer

// Dequantize weight
int16_t int16_weight = weight_int16[i];
float float_weight = (float)int16_weight * pow(2.0, -weight_q);

// Dequantize bias  
int16_t int16_bias = bias_int16[i];
float float_bias = (float)int16_bias * pow(2.0, -bias_q);

// Dequantize input feature map
int16_t int16_input = input_feature_map[i];
float float_input = (float)int16_input * pow(2.0, -input_q);

// Quantize output feature map
float float_output = compute_output(float_input, float_weight, float_bias);
int16_t int16_output = (int16_t)(float_output * pow(2.0, output_q));
```

## Examples

### YOLOv2 Extraction

**Float32 extraction:**
```bash
cd cpp
./weights_extractor \
    --cfg ../cfg/yolov2.cfg \
    --weights ../weights/yolov2.weights
```

Output files will be in `outputs/` directory: `outputs/weights.bin` and `outputs/bias.bin`

**With INT16 quantization:**
```bash
cd cpp
./weights_extractor \
    --cfg ../cfg/yolov2.cfg \
    --weights ../weights/yolov2.weights \
    --int16 \
    --verbose
```

### Custom Keras Model

```bash
cd python
python h5_to_darknet.py \
    --input my_model.h5 \
    --output-weights my_model.weights \
    --output-cfg my_model.cfg \
    --img-size 512 \
    --verbose

cd ../cpp
./weights_extractor \
    --cfg my_model.cfg \
    --weights my_model.weights \
    --int16
```

This will generate both float32 and INT16 quantized outputs for hardware deployment.

### Batch Processing

```bash
# Extract multiple models
cd cpp
for model in yolov2 yolov3 custom; do
    ./weights_extractor \
        --cfg ../models/${model}.cfg \
        --weights ../models/${model}.weights \
        --output-weights ../output/${model}_weights.bin \
        --output-bias ../output/${model}_bias.bin
done
```

## Supported Architectures

- **YOLOv2**: Fully tested and validated (float32 and INT16 quantization)
- **YOLOv3**: Fully supported (float32 and INT16 quantization, see [YOLOv3 Guide](YOLOV3_GUIDE.md))
- **Custom CNNs**: Any Keras model with Conv2D + BatchNormalization
- **Non-standard architectures**: Grouped layers, custom patterns supported

The tool uses intelligent name-based matching to handle various layer arrangements. INT16 quantization works with all supported architectures, automatically selecting optimal Q values per layer for each model.

## Project Structure

```
weight_converter_tool/
├── python/                    # Python H5 converter
│   ├── h5_to_darknet.py      # Main converter script
│   ├── model_parsers/        # Framework-specific parsers
│   ├── darknet_writer.py     # Darknet format writer
│   ├── cfg_generator.py      # Configuration generator
│   └── requirements.txt      # Python dependencies
│
├── cpp/                       # C++ binary extractor
│   ├── src/                  # Source files
│   │   ├── int16_quantizer.* # INT16 quantization module
│   ├── Makefile              # Build system
│   └── weights_extractor     # Compiled binary
│
├── examples/                  # Usage examples
│   ├── convert_yolov2.sh
│   └── convert_custom.sh
│
├── docs/                      # Additional documentation
├── README.md                  # This file
└── QUICKSTART.md             # Quick start guide
```

## Performance

**Python Converter:**
- Small models (<100 layers): <1 second
- Large models (YOLOv3, 106 layers): 2-5 seconds

**C++ Extractor:**
- YOLOv2 (50M params): <1 second (float32), <1.5 seconds (with INT16 quantization)
- YOLOv3 (62M params): ~2 seconds (float32), ~2.5 seconds (with INT16 quantization)
- Memory efficient: Streams data, minimal RAM usage
- INT16 quantization adds minimal overhead while providing significant memory savings

## Troubleshooting

### H5 Loading Fails

**Issue**: Model cannot be loaded from H5 file

**Solution**: The H5 file may contain only weights (not full model). Options:
1. Re-save with `model.save()` instead of `model.save_weights()`
2. Use the original training script to build the architecture first
3. If you have existing .weights files, skip Python step and use C++ extractor directly

See [YOLOv3 Guide](YOLOV3_GUIDE.md) for details.

### Size Mismatches

**Issue**: Output file sizes don't match expectations

**Solution**: The tool extracts only convolutional layers with BN folding applied. This is expected and correct. Use `--verbose` to see layer-by-layer details.

### Compilation Errors

**Issue**: C++ compilation fails

**Solution**: 
```bash
cd cpp
make clean
make
```

Ensure C++11 or newer compiler is available.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

- Inspired by the [Darknet](https://github.com/pjreddie/darknet) framework
- YOLOv2/v3 architecture from Joseph Redmon's work
- Built for hardware acceleration research

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{weight_converter_tool,
  title = {Neural Network Weight Extractor},
  author = {Solomon Negussie Tesema},
  email = {solomon.negussie.tesema@gmail.com},
  year = {2025},
  url = {https://github.com/yourusername/weight_converter_tool}
}
```

## License

See [LICENSE](LICENSE.md)

## Support

- Documentation: See [docs/](docs/) folder
- Quick Start: [QUICKSTART.md](QUICKSTART.md)
- YOLOv3 Guide: [YOLOV3_GUIDE.md](YOLOV3_GUIDE.md)
- Issues: GitHub Issues page
- Discussions: GitHub Discussions

---

**Ready for**: FPGA implementation, ASIC design, embedded systems, custom accelerators, and edge AI deployment. With INT16 quantization support, the tool provides optimized weight formats for resource-constrained hardware platforms.

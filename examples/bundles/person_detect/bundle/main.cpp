/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <inttypes.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <string>
#include <vector>

#include "person_detect_quant.h"

/// This is an example demonstrating how to use auto-generated bundles and
/// create standalone executables that can perform neural network computations.
/// This example loads and runs the compiled vww_96_int8 network model.
/// This example is using the static bundle API.

#define DEFAULT_HEIGHT 96
#define DEFAULT_WIDTH 96
#define CHANEL 1
#define OUTPUT_LEN 2

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
std::vector<std::string> inputImageFilenames;

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// Stores the smallest ex. time
void store_time(double new_time){
  double min_time;
  FILE *fp;
  fp = fopen("execution_time","r+");
  fscanf(fp, "%lf\n", &min_time);
  fclose(fp);
  printf("%lf\n",min_time);
  printf("%lf\n",new_time);
  if (new_time < min_time){
    fp = fopen("execution_time","w+");
    fprintf(fp, "%lf\n", new_time);
    fclose(fp);
  }
}

/// Reads a PNG image from a file into a newly allocated memory block \p imageT
/// representing a WxHxNxC tensor and returns it. The client is responsible for
/// freeing the memory block.
bool readPngImage(const char *filename, std::pair<float, float> range,
                  float *&imageT, size_t *imageDims) {
  unsigned char header[8];
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  // Can't open the file.
  if (!fp) {
    return true;
  }

  // Validate signature.
  size_t fread_ret = fread(header, 1, 8, fp);
  if (fread_ret != 8) {
    return true;
  }
  if (png_sig_cmp(header, 0, 8)) {
    return true;
  }

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    return true;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    return true;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  size_t width = png_get_image_width(png_ptr, info_ptr);
  size_t height = png_get_image_height(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;
  const size_t numChannels = 3;

  (void)bit_depth;
  assert(bit_depth == 8 && "Invalid image");
  bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

  int number_of_passes = png_set_interlace_handling(png_ptr);
  (void)number_of_passes;
  assert(number_of_passes == 1 && "Invalid image");

  png_read_update_info(png_ptr, info_ptr);

  // Error during image read.
  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, info_ptr);

  imageDims[0] = width;
  imageDims[1] = height;
  imageDims[2] = numChannels;
  imageT = static_cast<float *>(
      calloc(1, width * height * numChannels * sizeof(float)));

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t row_n = 0; row_n < height; row_n++) {
    png_byte *row = row_pointers[row_n];
    for (size_t col_n = 0; col_n < width; col_n++) {
      png_byte *ptr =
          &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
      imageT[getXYZ(imageDims, row_n, col_n, 0)] = float(ptr[0]) * scale + bias;
    }
  }

  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);
  printf("Loaded image: %s\n", filename);

  return false;
}

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 1x28x28 format.
static void loadImagesAndPreprocess(const std::vector<std::string> &filenames,
                                    float *&resultT, size_t *resultDims) {
  assert(filenames.size() > 0 &&
         "There must be at least one filename in filenames");
  std::pair<float, float> range = std::make_pair(0., 1.0);
  unsigned numImages = filenames.size();
  // N x C x H x W
  resultDims[0] = numImages;
  resultDims[1] = 3;
  resultDims[2] = DEFAULT_HEIGHT;
  resultDims[3] = DEFAULT_WIDTH;
  size_t resultSizeInBytes =
      numImages * DEFAULT_HEIGHT * DEFAULT_WIDTH * sizeof(float);
  resultT = static_cast<float *>(malloc(resultSizeInBytes));
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
  for (unsigned n = 0; n < numImages; n++) {
    float *imageT{nullptr};
    size_t dims[3];
    bool loadSuccess = !readPngImage(filenames[n].c_str(), range, imageT, dims);
    assert(loadSuccess && "Error reading input image.");
    (void)loadSuccess;

    assert((dims[0] == DEFAULT_HEIGHT && dims[1] == DEFAULT_WIDTH) &&
           "All images must have the same Height and Width");

    // Convert to BGR, as this is what NN is expecting.
    for (unsigned y = 0; y < dims[1]; y++) {
      for (unsigned x = 0; x < dims[0]; x++) {
        resultT[getXYZW(resultDims, n, 0, x, y)] =
            imageT[getXYZ(dims, x, y, 0)];
      }
    }
  }
  printf("Loaded images size in bytes is: %lu\n", resultSizeInBytes);
}

/// Parse images file names into a vector.
void parseCommandLineOptions(int argc, char **argv) {
  int arg = 1;
  while (arg < argc) {
    inputImageFilenames.push_back(argv[arg++]);
  }
}


/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void printResults(int8_t *results) {
  int maxIdx = 0;
  float maxValue = -1.0;
  for (int i = 0; i < OUTPUT_LEN; ++i) {
    printf("%d == %d\n",i,results[i]);
    if ((0.00390625*(results[i]+128)) > maxValue) {
      maxValue = (0.00390625*(results[i]+128));
      maxIdx = i;
    }
  }
  printf("Result: %d\n", maxIdx);
  printf("Confidence: %f\n", maxValue);
}

int main(int argc, char **argv) {
  clock_t start, end;
  assert(argc == 2 && "Usage: vww_96_int8 <cat.bin>");

  start = clock();
  //===----------------------------------------------------------------------===//
  //                 Wrapper code for executing a bundle
  //===----------------------------------------------------------------------===//
  /// Statically allocate memory for constant weights (model weights) and
  /// initialize.
  GLOW_MEM_ALIGN(PERSON_DETECT_QUANT_MEM_ALIGN)
  uint8_t constantWeight[PERSON_DETECT_QUANT_CONSTANT_MEM_SIZE] = {
  #include "person_detect_quant.weights.txt"
  };

  /// Statically allocate memory for mutable weights (model input/output data).
  GLOW_MEM_ALIGN(PERSON_DETECT_QUANT_MEM_ALIGN)
  uint8_t mutableWeight[PERSON_DETECT_QUANT_MUTABLE_MEM_SIZE];

  /// Statically allocate memory for activations (model intermediate results).
  GLOW_MEM_ALIGN(PERSON_DETECT_QUANT_MEM_ALIGN)
  uint8_t activations[PERSON_DETECT_QUANT_ACTIVATIONS_MEM_SIZE];

  /// Bundle input data absolute address.
  int8_t *inputAddr = (int8_t *)GLOW_GET_ADDR(mutableWeight, PERSON_DETECT_QUANT_input);

  /// Bundle output data absolute address.
  uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, PERSON_DETECT_QUANT_MobilenetV1_Predictions_Reshape_1);

  // Initialize input images.
  //initInputImages(argv[1]);
  int8_t inputT[DEFAULT_HEIGHT * DEFAULT_WIDTH * CHANEL];
  //loadImagesAndPreprocess(..,..)
  FILE* fp = fopen(argv[1], "rb");
  (void)fread(inputT, 1 * CHANEL * DEFAULT_HEIGHT * DEFAULT_WIDTH, 1 /*bytes*/, fp); 
  fclose(fp);

  memcpy(inputAddr, inputT, DEFAULT_HEIGHT * DEFAULT_WIDTH * CHANEL * sizeof(int8_t));
  end = clock(); // End timer
  double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Initialization time: %lf\n",cpu_time_used);

  // Perform the computation.
  start = clock();
  //run sample
  int errCode = person_detect_quant(constantWeight, mutableWeight, activations);
  // End timer
  end = clock();
  if (errCode != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode);
  }

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Execution time: %lf\n",cpu_time_used);
  //store_time(cpu_time_used);

  int8_t *results = (int8_t *)(outputAddr);
  // Print results.
  printResults(results);
  return 0;
}


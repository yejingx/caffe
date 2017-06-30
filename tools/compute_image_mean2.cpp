#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;

DEFINE_int32(width, 256, "Image Width");
DEFINE_int32(height, 256, "Image Height");
DEFINE_int32(channel, 3, "Image Channel");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a file list\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] files.txt [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  std::ifstream infile(argv[1]);
  std::string line;

  Datum datum;
  BlobProto sum_blob;
  int count = 0;

  sum_blob.set_num(1);
  sum_blob.set_channels(FLAGS_channel);
  sum_blob.set_height(FLAGS_height);
  sum_blob.set_width(FLAGS_width);
  const int data_size = FLAGS_channel * FLAGS_height * FLAGS_width;
  int size_in_datum = std::max<int>(data_size,
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting iteration";

  bool status;
  size_t pos;
  int label;
  std::string path;

  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    path = line.substr(0, pos);
    label = atoi(line.substr(pos + 1).c_str());
    status = ReadImageToDatum(path, label, FLAGS_height, FLAGS_width, true,
        "", &datum);
    if (status == false) continue;

    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum << ", data_size: " << data_size;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]: " << mean_values[c] / dim;
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}

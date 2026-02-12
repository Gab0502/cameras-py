#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

namespace py = pybind11;
namespace fs = std::filesystem;

class FastVideoMerger {
private:
    std::string streams_dir;
    std::string recordings_dir;

public:
    FastVideoMerger(const std::string& streams, const std::string& recordings)
        : streams_dir(streams), recordings_dir(recordings) {
        av_log_set_level(AV_LOG_ERROR);
    }

    // Lista arquivos .ts de uma câmera específica
    std::vector<std::string> get_ts_files(const std::string& camera_id, int last_seconds = -1) {
        std::vector<std::string> files;
        std::string prefix = "camera_" + camera_id + "_";

        for (const auto& entry : fs::directory_iterator(streams_dir)) {
            if (entry.path().extension() == ".ts") {
                std::string filename = entry.path().filename().string();
                if (filename.rfind(prefix, 0) == 0) {
                    files.push_back(entry.path().string());
                }
            }
        }

        std::sort(files.begin(), files.end());

        // Filtra últimos N segundos (assumindo segmentos de 2s)
        if (last_seconds > 0) {
            int hls_time = 2;
            int count = std::max(1, last_seconds / hls_time);
            if ((int)files.size() > count) {
                files.erase(files.begin(), files.end() - count);
            }
        }

        return files;
    }

    // Junta segmentos .ts em um .mp4
    std::string merge_segments(const std::string& camera_id, int last_seconds = -1) {
        auto ts_files = get_ts_files(camera_id, last_seconds);
        if (ts_files.empty()) {
            return "";
        }

        // Gera nome do arquivo de saída
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "camera_" << camera_id << "_"
           << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S")
           << ".mp4";
        std::string output_file = recordings_dir + "/" + ss.str();

        // Cria lista de arquivos temporária
        std::string list_file = streams_dir + "/temp_list_" + camera_id + ".txt";
        std::ofstream list(list_file);
        for (const auto& file : ts_files) {
            list << "file '" << fs::absolute(file).string() << "'\n";
        }
        list.close();

        // Contextos FFmpeg
        AVFormatContext* input_ctx = nullptr;
        AVFormatContext* output_ctx = nullptr;
        int ret = 0;

        // Abre lista concat
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "safe", "0", 0);

        ret = avformat_open_input(&input_ctx, list_file.c_str(),
                                  av_find_input_format("concat"), &opts);
        if (ret < 0) {
            av_dict_free(&opts);
            fs::remove(list_file);
            return "";
        }

        avformat_find_stream_info(input_ctx, nullptr);

        // Cria contexto de saída
        avformat_alloc_output_context2(&output_ctx, nullptr, nullptr, output_file.c_str());
        if (!output_ctx) {
            avformat_close_input(&input_ctx);
            av_dict_free(&opts);
            fs::remove(list_file);
            return "";
        }

        // Copia streams
        for (unsigned int i = 0; i < input_ctx->nb_streams; i++) {
            AVStream* in_stream = input_ctx->streams[i];
            AVStream* out_stream = avformat_new_stream(output_ctx, nullptr);
            if (!out_stream) continue;

            avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
            out_stream->codecpar->codec_tag = 0;
        }

        // Abre arquivo de saída
        if (!(output_ctx->oformat->flags & AVFMT_NOFILE)) {
            ret = avio_open(&output_ctx->pb, output_file.c_str(), AVIO_FLAG_WRITE);
            if (ret < 0) {
                avformat_free_context(output_ctx);
                avformat_close_input(&input_ctx);
                av_dict_free(&opts);
                fs::remove(list_file);
                return "";
            }
        }

        // Escreve header
        AVDictionary* output_opts = nullptr;
        av_dict_set(&output_opts, "movflags", "+faststart", 0);
        ret = avformat_write_header(output_ctx, &output_opts);
        av_dict_free(&output_opts);

        if (ret < 0) {
            if (!(output_ctx->oformat->flags & AVFMT_NOFILE))
                avio_closep(&output_ctx->pb);
            avformat_free_context(output_ctx);
            avformat_close_input(&input_ctx);
            av_dict_free(&opts);
            fs::remove(list_file);
            return "";
        }

        // Copia pacotes sem reencode
        AVPacket pkt;
        while (av_read_frame(input_ctx, &pkt) >= 0) {
            AVStream* in_stream = input_ctx->streams[pkt.stream_index];
            AVStream* out_stream = output_ctx->streams[pkt.stream_index];

            av_packet_rescale_ts(&pkt, in_stream->time_base, out_stream->time_base);
            pkt.pos = -1;

            ret = av_interleaved_write_frame(output_ctx, &pkt);
            av_packet_unref(&pkt);

            if (ret < 0) break;
        }

        // Finaliza
        av_write_trailer(output_ctx);

        if (output_ctx && !(output_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&output_ctx->pb);
        }

        avformat_free_context(output_ctx);
        avformat_close_input(&input_ctx);
        av_dict_free(&opts);
        fs::remove(list_file);

        return output_file;
    }

    // Versão paralela para múltiplas câmeras
    std::map<std::string, std::string> merge_multiple(const std::vector<std::string>& camera_ids, int last_seconds = -1) {
        std::map<std::string, std::string> results;

        #pragma omp parallel for
        for (size_t i = 0; i < camera_ids.size(); i++) {
            const std::string& cam_id = camera_ids[i];
            std::string result = merge_segments(cam_id, last_seconds);

            #pragma omp critical
            {
                results[cam_id] = result;
            }
        }

        return results;
    }
};

// Bindings Python
PYBIND11_MODULE(fast_video_merger, m) {
    py::class_<FastVideoMerger>(m, "FastVideoMerger")
        .def(py::init<const std::string&, const std::string&>())
        .def("merge_segments", &FastVideoMerger::merge_segments,
             py::arg("camera_id"),
             py::arg("last_seconds") = -1)
        .def("merge_multiple", &FastVideoMerger::merge_multiple,
             py::arg("camera_ids"),
             py::arg("last_seconds") = -1);
}

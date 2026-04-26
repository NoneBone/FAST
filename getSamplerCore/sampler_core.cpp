#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>
#include <math.h>
// added by fast
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef int NodeIDType;
typedef int EdgeIDType;
typedef float TimeStampType;
#define COMMON 1 // shell edit or manual edit
#define UNI_FLAG COMMON
#define INV_FLAG COMMON
#define CNT_OR_INV COMMON
#define EDGE_INV 0
#define NODE_CNT (EDGE_INV ^ 1)
class TemporalGraphBlock
{
    public:
        std::vector<NodeIDType> row;
        std::vector<NodeIDType> col;
        std::vector<EdgeIDType> eid;
        std::vector<TimeStampType> ts;
        std::vector<TimeStampType> dts;
        std::vector<NodeIDType> nodes;
        // std::vector<NodeIDType> uniN;
        // std::vector<NodeIDType> invN;
        std::vector<EdgeIDType> uniE;
        std::vector<EdgeIDType> invE;
        std::vector<EdgeIDType> cntE;
        NodeIDType dim_in, dim_out;
        double ptr_time = 0;
        double search_time = 0;
        double sample_time = 0;
        double tot_time = 0;
        double coo_time = 0;
        int uniFlag = 1;
        int invFlag = 1;
        int cntFlag = 1;

        TemporalGraphBlock(){}

        TemporalGraphBlock(std::vector<NodeIDType> &_row, std::vector<NodeIDType> &_col,
                           std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                           std::vector<TimeStampType> &_dts, std::vector<NodeIDType> &_nodes, 
                           std::vector<EdgeIDType> &_uniE, std::vector<EdgeIDType> &_invE, std::vector<EdgeIDType> &_cntE,
                           NodeIDType _dim_in, NodeIDType _dim_out) :
                           row(_row), col(_col), eid(_eid), ts(_ts), dts(_dts),
                           nodes(_nodes), uniE(_uniE), invE(_invE), cntE(_cntE),
                           dim_in(_dim_in), dim_out(_dim_out) {}
};

class ParallelSampler
{
    public:
        std::vector<EdgeIDType> indptr;
        std::vector<EdgeIDType> indices;
        std::vector<EdgeIDType> eid;
        std::vector<TimeStampType> ts;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int num_thread_per_worker;
        int num_workers;
        int num_threads;
        int num_layers;
        std::vector<int> num_neighbors;
        bool recent;
        bool prop_time;
        int num_history;
        TimeStampType window_duration;
        std::vector<std::vector<std::vector<EdgeIDType>::size_type>> ts_ptr;
        omp_lock_t *ts_ptr_lock;
        std::vector<TemporalGraphBlock> ret;

        ParallelSampler(std::vector<EdgeIDType> &_indptr, std::vector<EdgeIDType> &_indices,
                        std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                        int _num_thread_per_worker, int _num_workers, int _num_layers,
                        std::vector<int> &_num_neighbors, bool _recent, bool _prop_time,
                        int _num_history, TimeStampType _window_duration) :
                        indptr(_indptr), indices(_indices), eid(_eid), ts(_ts), prop_time(_prop_time),
                        num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
                        num_layers(_num_layers), num_neighbors(_num_neighbors), recent(_recent),
                        num_history(_num_history), window_duration(_window_duration)
        {
            omp_set_num_threads(num_thread_per_worker * num_workers);
            num_threads = num_thread_per_worker * num_workers;
            num_nodes = indptr.size() - 1;
            num_edges = indices.size();
            ts_ptr_lock = (omp_lock_t *)malloc(num_nodes * sizeof(omp_lock_t)); 
            for (int i = 0; i < num_nodes; i++) 
                omp_init_lock(&ts_ptr_lock[i]);
            ts_ptr.resize(num_history + 1);
            for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
            {
                it->resize(indptr.size() - 1);
#pragma omp parallel for
                for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                    (*it)[itt - indptr.begin()] = *itt;
            }
        }// After initialization is completed, the following results are obtained: for each graph node, the lock and the 2 ts_ptr are placed at the number of 1-hop neighbors of each uni node.

    ~ParallelSampler() {
        for (int i = 0; i < num_nodes; i++)
            omp_destroy_lock(&ts_ptr_lock[i]);
        free(ts_ptr_lock);
    }

        void reset()
        {
            for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
            {
                it->resize(indptr.size() - 1);
#pragma omp parallel for
                for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                    (*it)[itt - indptr.begin()] = *itt;
            }
        }

        void update_ts_ptr(int slc, std::vector<NodeIDType> &root_nodes, 
                           std::vector<TimeStampType> &root_ts, float offset)
        {// Update ts_ptr[1] and combine it with ts_ptr[0] to represent all the neighbors that meet the time constraints
#pragma omp parallel for schedule(static, int(ceil(static_cast<float>(root_nodes.size()) / num_threads))) // Static scheduling, each thread is allocated a number of loop iterations equal to chunk_size
            for (std::vector<NodeIDType>::size_type i = 0; i < root_nodes.size(); i++)
            {
                NodeIDType n = root_nodes[i];
                omp_set_lock(&(ts_ptr_lock[n]));
                for (std::vector<EdgeIDType>::size_type j = ts_ptr[slc][n]; j < indptr[n + 1]; j++)
                {
                    // std::cout << "comparing " << ts[j] << " with " << root_ts[i] << std::endl;
                    if (ts[j] > (root_ts[i] + offset - 1e-7f))
                    {
                        if (j != ts_ptr[slc][n])
                            ts_ptr[slc][n] = j - 1;
                        break;
                    }
                    if (j == indptr[n + 1] - 1)// The neighbors that existed before the birth time of the root node all meet the requirements.
                    {
                        ts_ptr[slc][n] = j;
                    }
                }
                omp_unset_lock(&(ts_ptr_lock[n]));
            }
        }

        inline void add_neighbor(std::vector<NodeIDType> *_row, std::vector<NodeIDType> *_col,
                                 std::vector<EdgeIDType> *_eid, std::vector<TimeStampType> *_ts,
                                 std::vector<TimeStampType> *_dts, std::vector<NodeIDType> *_nodes, 
                                 EdgeIDType &k, TimeStampType &src_ts, int &row_id)
        {
            _row->push_back(row_id);
            _col->push_back(_nodes->size());
            _eid->push_back(eid[k]);
            if (prop_time)
                _ts->push_back(src_ts);
            else
                _ts->push_back(ts[k]);
            _dts->push_back(src_ts - ts[k]);
            _nodes->push_back(indices[k]);
            // _row.push_back(0);
            // _col.push_back(0);
            // _eid.push_back(0);
            // if (prop_time)
            //     _ts.push_back(src_ts);
            // else
            //     _ts.push_back(10000);
            // _nodes.push_back(100);
        }
        int64_t opt_hash_s(int32_t s) {
            return static_cast<int64_t>(s);
        }
inline void combine_coo(TemporalGraphBlock &_ret, 
                std::vector<NodeIDType> **_row,
                std::vector<NodeIDType> **_col, 
                std::vector<EdgeIDType> **_eid, 
                std::vector<TimeStampType> **_ts, 
                std::vector<TimeStampType> **_dts,
                std::vector<NodeIDType> **_nodes,
                std::vector<int> &_out_nodes)
{
    std::vector<EdgeIDType> cum_row, cum_col;
    cum_row.reserve(num_threads + 1);
    cum_col.reserve(num_threads + 1);
    cum_row.push_back(0);
    cum_col.push_back(0);
    
    for (int tid = 0; tid < num_threads; tid++)
    {
        cum_row.push_back(cum_row.back() + _out_nodes[tid]);
        cum_col.push_back(cum_col.back() + _col[tid]->size());
    }
    
    int num_root_nodes = _ret.nodes.size();// 6000
    size_t total_edges = cum_col.back();// 1882
    
    _ret.row.reserve(total_edges);
    _ret.col.reserve(total_edges);
    _ret.eid.reserve(total_edges);
    _ret.ts.reserve(total_edges + num_root_nodes);
    _ret.dts.reserve(total_edges + num_root_nodes);
    _ret.nodes.reserve(total_edges + num_root_nodes);
    
    _ret.row.resize(total_edges);
    _ret.col.resize(total_edges);
    _ret.eid.resize(total_edges);
    _ret.ts.resize(total_edges + num_root_nodes);
    _ret.dts.resize(total_edges + num_root_nodes);
    _ret.nodes.resize(total_edges + num_root_nodes);
    
#if UNI_FLAG
#if INV_FLAG
    std::vector<int32_t> inv_idx(total_edges, -1);
#endif
    std::vector<std::unordered_map<EdgeIDType, ssize_t>> local_key2idx(num_threads);
    std::vector<std::vector<EdgeIDType>> local_uniq_eid(num_threads);
    
    std::vector<EdgeIDType> all_eids;
    all_eids.reserve(total_edges);
#endif

#if CNT_OR_INV
    #if EDGE_INV

    std::vector<NodeIDType> cntE_vec;
    cntE_vec.reserve(total_edges);
    # else
    std::vector<int> row_counts;
    std::vector<NodeIDType> unique_rows;
    std::vector<int> cntE_vec;

    row_counts.reserve(total_edges);
    unique_rows.reserve(total_edges);
    cntE_vec.reserve(total_edges + 1);
    cntE_vec.push_back(0);
    # endif
#endif
    
#pragma omp parallel for schedule(static, 1)
    for (int tid = 0; tid < num_threads; tid++)
    {
        auto* src_row = _row[tid]->data();
        auto* src_col = _col[tid]->data();
        auto* src_eid = _eid[tid]->data();
        auto* src_ts = _ts[tid]->data();
        auto* src_dts = _dts[tid]->data();
        auto* src_nodes = _nodes[tid]->data();
        
        auto* dst_row = _ret.row.data() + cum_col[tid];
        auto* dst_col = _ret.col.data() + cum_col[tid];
        auto* dst_eid = _ret.eid.data() + cum_col[tid];
        auto* dst_ts = _ret.ts.data() + cum_col[tid] + num_root_nodes;
        auto* dst_dts = _ret.dts.data() + cum_col[tid] + num_root_nodes;
        auto* dst_nodes = _ret.nodes.data() + cum_col[tid] + num_root_nodes;
        
        size_t local_size = _col[tid]->size();
        
        NodeIDType row_offset = cum_row[tid];
        NodeIDType col_offset = cum_col[tid] + num_root_nodes;
        
        for (size_t i = 0; i < local_size; ++i) {
            dst_row[i] = src_row[i] + row_offset;
            dst_col[i] = src_col[i] + col_offset;
            dst_eid[i] = src_eid[i];
        }
        
        size_t data_size = _ts[tid]->size();
        for (size_t i = 0; i < data_size; ++i) {
            dst_ts[i] = src_ts[i];
            dst_dts[i] = src_dts[i];
            dst_nodes[i] = src_nodes[i];
        }
                
        #if UNI_FLAG
                if(_ret.uniFlag) {
                    local_key2idx[tid].reserve(local_size);
                    local_uniq_eid[tid].reserve(local_size);
                    
                    auto &local_map = local_key2idx[tid];
                    auto &local_vec = local_uniq_eid[tid];
                    
                    std::vector<EdgeIDType> thread_eids;
                    thread_eids.reserve(local_size);
                    for (size_t i = 0; i < local_size; ++i) {
                        thread_eids.push_back(src_eid[i]);
                    }
                    
        #pragma omp critical(all_eids_append)
                    {
                        all_eids.insert(all_eids.end(), 
                                    thread_eids.begin(), thread_eids.end());
                    }
                    for (size_t i = 0; i < local_size; i++) {
                        EdgeIDType eid_val = src_eid[i];
                        auto iter = local_map.find(eid_val);
                        if (iter == local_map.end()) {
                            local_map[eid_val] = local_vec.size();
                            local_vec.push_back(eid_val);
                        }
                    }
                }
        #endif
                
        #if CNT_OR_INV && EDGE_INV
                if (local_size > 0) {
                    std::vector<NodeIDType> thread_cntE;
                    thread_cntE.reserve(local_size);
                    
                    NodeIDType current_row = src_row[0];
                    int current_idx = 0;
                    thread_cntE.push_back(current_idx);
                    
                    for (size_t i = 1; i < local_size; i++) {
                        if (src_row[i] != current_row) {
                            current_row = src_row[i];
                            current_idx++;
                        }
                        thread_cntE.push_back(current_idx);
                    }
                    
        #pragma omp critical(cntE_append)
                    {
                        for (auto idx : thread_cntE) {
                            cntE_vec.push_back(idx);
                        }
                    }
                }
        #endif
                
                delete _row[tid];
                delete _col[tid];
                delete _eid[tid];
                delete _ts[tid];
                delete _dts[tid];
                delete _nodes[tid];
            }

        #if CNT_OR_INV
            # if EDGE_INV
            // the global inverse index
            if (total_edges > 0 && cntE_vec.size() == total_edges) {
                std::vector<int> global_cntE = std::move(cntE_vec);
                
                std::vector<NodeIDType> unique_rows;
                unique_rows.reserve(total_edges);
                const NodeIDType* row_data = _ret.row.data();
                
                NodeIDType current_row = row_data[0];
                unique_rows.push_back(current_row);
                
                for (size_t i = 1; i < total_edges; i++) {
                    if (row_data[i] != current_row) {
                        current_row = row_data[i];
                        unique_rows.push_back(current_row);
                    }
                }
                
                std::unordered_map<NodeIDType, int> row_to_global_idx;
                for (size_t i = 0; i < unique_rows.size(); i++) {
                    row_to_global_idx[unique_rows[i]] = i;
                }
                
                for (size_t i = 0; i < total_edges; i++) {
                    auto it = row_to_global_idx.find(row_data[i]);
                    if (it != row_to_global_idx.end()) {
                        global_cntE[i] = it->second;
                    }
                }
                
                _ret.cntE = std::move(global_cntE);
            #else
            if (total_edges > 0) {
                const NodeIDType* row_data = _ret.row.data();
                
                NodeIDType current_row = row_data[0];
                int count = 1;
                int cum_sum = 0;
                
                for (size_t i = 1; i < total_edges; i++) {
                    if (row_data[i] == current_row) {
                        count++;
                    } else {
                        row_counts.push_back(count);
                        unique_rows.push_back(current_row);
                        cum_sum += count;
                        cntE_vec.push_back(cum_sum);
                        
                        current_row = row_data[i];
                        count = 1;
                    }
                }
                row_counts.push_back(count);
                unique_rows.push_back(current_row);
                cum_sum += count;
                cntE_vec.push_back(cum_sum);
                
                _ret.cntE = std::move(cntE_vec);
            #endif
            }
        #endif
            
        #if UNI_FLAG
            if(_ret.uniFlag) {
                // deduplication
                std::unordered_map<EdgeIDType, ssize_t> global_eid_map;
                std::vector<EdgeIDType> global_uniq_eid;
                
                global_eid_map.reserve(all_eids.size());
                global_uniq_eid.reserve(all_eids.size());
                
                for (size_t i = 0; i < all_eids.size(); i++) {
                    EdgeIDType eid_val = all_eids[i];
                    auto iter = global_eid_map.find(eid_val);
                    if (iter == global_eid_map.end()) {
                        ssize_t new_index = global_uniq_eid.size();
                        global_eid_map[eid_val] = new_index;
                        global_uniq_eid.push_back(eid_val);
                    }
                }
                
        #if INV_FLAG
                // inv_idx
                const EdgeIDType* eid_data = _ret.eid.data();
                int32_t* inv_data = inv_idx.data();
                
                for (size_t i = 0; i < total_edges; i++) {
                    auto it = global_eid_map.find(eid_data[i]);
                    if (it != global_eid_map.end()) {
                        inv_data[i] = static_cast<int32_t>(it->second);
                    }
                }
        #endif

                _ret.uniE = std::move(global_uniq_eid);
                
        #if INV_FLAG
                _ret.invE = std::move(inv_idx);
        #endif
            }
        #endif
            
            _ret.dim_in = _ret.nodes.size();
            _ret.dim_out = cum_row.back();
        }

        void sample_layer(std::vector<NodeIDType> &_root_nodes, std::vector<TimeStampType> &_root_ts,
                          int neighs, bool use_ptr, bool from_root)
        {
            double t_s = omp_get_wtime();
            std::vector<NodeIDType> *root_nodes;
            std::vector<TimeStampType> *root_ts;
            if (from_root)
            {
                root_nodes = &_root_nodes;
                root_ts = &_root_ts;
            }
            else {
                // add a else to initialize them, fix recent-sample-segmentFault in py3.10
                root_nodes = &(ret[ret.size() - 1 - num_history].nodes);
                root_ts = &(ret[ret.size() - 1 - num_history].ts);
            }
            double t_ptr_s = omp_get_wtime();
            if (use_ptr)
                update_ts_ptr(num_history, *root_nodes, *root_ts, 0);
            ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
            for (int i = 0; i < num_history; i++)
            {
                if (!from_root)
                {
                    root_nodes = &(ret[ret.size() - 1 - i - num_history].nodes);
                    root_ts = &(ret[ret.size() - 1 - i - num_history].ts);
                }
                TimeStampType offset = -i * window_duration;
                t_ptr_s = omp_get_wtime();
                if ((use_ptr) && (std::abs(window_duration) > 1e-7f))
                    update_ts_ptr(num_history - 1 - i, *root_nodes, *root_ts, offset - window_duration);
                ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
                std::vector<NodeIDType> *_row[num_threads];
                std::vector<NodeIDType> *_col[num_threads];
                std::vector<EdgeIDType> *_eid[num_threads];
                std::vector<TimeStampType> *_ts[num_threads];
                std::vector<TimeStampType> *_dts[num_threads];
                std::vector<NodeIDType> *_nodes[num_threads];
                std::vector<int> _out_node(num_threads, 0);
                int reserve_capacity = int(ceil((*root_nodes).size() / num_threads)) * neighs;
#pragma omp parallel
            {
                    int tid = omp_get_thread_num();
                    unsigned int loc_seed = tid;
                    _row[tid] = new std::vector<NodeIDType>;
                    _col[tid] = new std::vector<NodeIDType>;
                    _eid[tid] = new std::vector<EdgeIDType>;
                    _ts[tid] = new std::vector<TimeStampType>;
                    _dts[tid] = new std::vector<TimeStampType>;
                    _nodes[tid] = new std::vector<NodeIDType>;
                    _row[tid]->reserve(reserve_capacity);
                    _col[tid]->reserve(reserve_capacity);
                    _eid[tid]->reserve(reserve_capacity);
                    _ts[tid]->reserve(reserve_capacity);
                    _dts[tid]->reserve(reserve_capacity);
                    _nodes[tid]->reserve(reserve_capacity);
// #pragma omp critical
//                     std::cout<<tid<<" sampling: "<<root_nodes->size()<<" "<<int(ceil((*root_nodes).size() / num_threads))<<std::endl;
#pragma omp for schedule(static, int(ceil(static_cast<float>((*root_nodes).size()) / num_threads)))// static 500
                    for (std::vector<NodeIDType>::size_type j = 0; j < (*root_nodes).size(); j++)
                    {
                        NodeIDType n = (*root_nodes)[j];
                        TimeStampType nts = (*root_ts)[j];
                        EdgeIDType s_search, e_search;
                        if (use_ptr)
                        {
                            s_search = ts_ptr[num_history - 1 - i][n];// [0][n]
                            e_search = ts_ptr[num_history - i][n];// [1][n]
                        }
                        else
                        {
                            // search for start and end pointer
                            double t_search_s = omp_get_wtime();
                            if (num_history == 1)
                            {
                                // TGAT style
                                s_search = indptr[n];
                                auto e_it = std::upper_bound(ts.begin() + indptr[n], 
                                                             ts.begin() + indptr[n + 1], nts);
                                e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                            }
                            else
                            {
                                // DySAT style
                                auto s_it = std::upper_bound(ts.begin() + indptr[n],
                                                             ts.begin() + indptr[n + 1],
                                                             nts + offset - window_duration);
                                s_search = std::max(int(s_it - ts.begin()) - 1, indptr[n]);
                                auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                                             ts.begin() + indptr[n + 1], nts + offset);
                                e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                            }
                            if (tid == 0)
                                ret[0].search_time += omp_get_wtime() - t_search_s;
                        }
                        // std::cout << n << " " << s_search << " " << e_search << std::endl;
                        double t_sample_s = omp_get_wtime();
                        if ((recent) || (e_search - s_search < neighs)) // 
                        {                            
                            // no sampling, pick recent neighbors
                            for (EdgeIDType k = e_search; k > std::max(s_search, e_search - neighs); k--)
                            {
                                if (ts[k] < nts + offset - 1e-7f)
                                {
                                    add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid], _dts[tid], _nodes[tid],
                                              k, nts, _out_node[tid]);
                                }
                            }
                        }
                        else
                        {
                            // random sampling within ptr
                            for (int _i = 0; _i < neighs; _i++)
                            {
                                EdgeIDType picked = s_search + rand_r(&loc_seed) % (e_search - s_search + 1);
                                if (ts[picked] < nts + offset - 1e-7f)
                                {
                                    add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid], _dts[tid], _nodes[tid],
                                                    picked, nts, _out_node[tid]);
                                }
                            }
                        }
                        _out_node[tid] += 1;
                        if (tid == 0)
                            ret[0].sample_time += omp_get_wtime() - t_sample_s;
                    }
                }
                double t_coo_s = omp_get_wtime();
                ret[ret.size() - 1 - i].ts.insert(ret[ret.size() - 1 - i].ts.end(), 
                                                  root_ts->begin(), root_ts->end());
                ret[ret.size() - 1 - i].nodes.insert(ret[ret.size() - 1 - i].nodes.end(), 
                                                     root_nodes->begin(), root_nodes->end());
                ret[ret.size() - 1 - i].dts.resize(root_nodes->size());
                combine_coo(ret[ret.size() - 1 - i], _row, _col, _eid, _ts, _dts, _nodes, _out_node);
                ret[0].coo_time += omp_get_wtime() - t_coo_s;
            }
            ret[0].tot_time += omp_get_wtime() - t_s;
        }
        
        void sample(std::vector<NodeIDType> &root_nodes, std::vector<TimeStampType> &root_ts, std::vector<NodeIDType> &csrFlag)
        {
            // a weird bug, dgl library seems to modify the total number of threads
#define USE_SA_OMP 0
#if !USE_SA_OMP
            // omp_set_dynamic(1); // effective is not good
            // omp_set_num_threads(num_threads);
#else // USE_SA_OMP
            static int initialNum = 1;
            int max_thread_id = -1;

#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int num_threads_in_team = omp_get_num_threads();
                int local_max_id = num_threads_in_team - 1;
// Use the critical section to ensure that only one thread can update the max_thread_id
#pragma omp critical
                {
                    if (local_max_id > max_thread_id) {
                        max_thread_id = local_max_id;
                    }
                }
            }
            
            if(initialNum --> 0 || max_thread_id != num_threads)
                // printf("DGL OMP THREAD NUM IS: %d\n", max_thread_id);
                omp_set_num_threads(num_threads);// DGL will affect the OMP configuration
            static int* thread_array = new int[num_threads];
            
            static int firstLoop = 1;
            static int isHtTure = 1;
            if(firstLoop-->0){
                
                if(isHtTure == 1){
                // 1.manually set
                // int config[][8] = {{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}};
                // std::copy(std::begin(config[0]), std::end(config[0]), thread_array);
                    for(int ii=0;ii < num_threads; ii++){
                        thread_array[ii] = ii;
                    }
                }
                else{// If hyper-threading is not set, initialize according to the number of physical cores
                    for(int ii=0;ii < physical_cores; ii++){
                        thread_array[ii] = ii;
                    }
                }
            }
            int* tmpArray = new int[num_threads];
            int configChange = 0;
            
            // Detect whether the current core mapping has changed
            // int getNumOMP = omp_get_num_threads(); // not use
#pragma omp parallel
                {
                    int thread_id = omp_get_thread_num();
                    int core_id = sched_getcpu();
                    tmpArray[thread_id] = core_id; // false sharing but not a big deal
                }
            for(int ii=0;ii < num_threads; ii++){
                if(tmpArray[ii] != thread_array[ii]){
                    configChange = 1;
                    break;
                }
            }
            // Only rebind for the first time or when the configuration changes
            if(configChange==1 || firstLoop ==0){
#pragma omp parallel
                {
                    int thread_id = omp_get_thread_num();
                    
                    // calculate the binding strategy: adjacent threads are bound to the same physical core
                    int physical_core = thread_array[thread_id] ; // physical core number
                    // int ht_index = thread_id % 2;              // hyper-threading index (0 or 1)
                    
                    // ensure not to exceed the system core range
                    int target_core = physical_core % logical_cores;
                    
                    // set CPU affinity
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(target_core, &cpuset);
                    
                    // bind the current thread to the specified core
                    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
                        perror("pthread_setaffinity_np");
                    }
                }
            }
#endif
            ret.resize(0);
            bool first_layer = true;
            bool use_ptr = false;
            for (int i = 0; i < num_layers; i++)
            {
                ret.resize(ret.size() + num_history);
                if (csrFlag.size() >= 3) {
                    ret[0].uniFlag = csrFlag[0];
                    ret[0].invFlag = csrFlag[1];
                    ret[0].cntFlag = csrFlag[2];
                } else {
                    // std::cout << "Error: csrFlag size must be at least 3" << std::endl;
                }
                if ((first_layer) || ((prop_time) && num_history == 1) || (recent))
                {
                    first_layer = false;
                    use_ptr = true;
                }
                else
                    use_ptr = false;
                if (i==0)
                    sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, true);
                else
                    sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, false);
            }
        }

        struct CoreInfo {
            int logical_cores;
            int physical_cores;
            bool is_hyperthreading;
            int numa_nodes;
        };
        struct NumaNodeRange {
            int node_index;
            int start_cpu;
            int end_cpu;
        };

        std::vector<NumaNodeRange> parseNumaCpuRanges(const std::string& cpu_list, int node_index) {
            std::vector<NumaNodeRange> result;
            std::stringstream ss(cpu_list);
            std::string item;
            
            // Separated by commas
            while (std::getline(ss, item, ',')) {
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
                
                if (item.empty()) continue;
                
                size_t dash_pos = item.find('-');
                if (dash_pos != std::string::npos) {
                    // For example, "0-19
                    int start = std::stoi(item.substr(0, dash_pos));
                    int end = std::stoi(item.substr(dash_pos + 1));
                    
                    NumaNodeRange range;
                    range.node_index = node_index;
                    range.start_cpu = start;
                    range.end_cpu = end;
                    
                    result.push_back(range);
                } else {
                    NumaNodeRange range;
                    range.node_index = node_index;
                    range.start_cpu = std::stoi(item);
                    range.end_cpu = 1;
                    
                    result.push_back(range);
                }
            }
            
            return result;
        }

        std::vector<NumaNodeRange> getNumaNodeStruct() {
            std::vector<NumaNodeRange> numa_ranges = {};
            
            // Parse the lscpu output to obtain the CPU information of the NUMA node
            FILE* lscpu_pipe = popen("lscpu", "r");
            if (lscpu_pipe) {
                char buffer[256];
                std::string numa_node0_cpus, numa_node1_cpus;
                
                while (fgets(buffer, sizeof(buffer), lscpu_pipe) != nullptr) {
                    std::string line(buffer);
                    
                    // Search for NUMA node0 CPU information
                    if (line.find("NUMA node0 CPU(s)") != std::string::npos) {
                        size_t pos = line.find(':');
                        if (pos != std::string::npos) {
                            numa_node0_cpus = line.substr(pos + 1);
                            numa_node0_cpus.erase(0, numa_node0_cpus.find_first_not_of(" \t"));
                            numa_node0_cpus.erase(numa_node0_cpus.find_last_not_of(" \t") + 1);
                        }
                    }
                    // Search for NUMA node1 CPU information
                    else if (line.find("NUMA node1 CPU(s)") != std::string::npos) {
                        size_t pos = line.find(':');
                        if (pos != std::string::npos) {
                            numa_node1_cpus = line.substr(pos + 1);
                            numa_node1_cpus.erase(0, numa_node1_cpus.find_first_not_of(" \t"));
                            numa_node1_cpus.erase(numa_node1_cpus.find_last_not_of(" \t") + 1);
                        }
                    }
                }
                pclose(lscpu_pipe);
                
                // Convert to NumaNodeRange
                if (!numa_node0_cpus.empty()) {
                    std::vector<NumaNodeRange> node0_ranges = parseNumaCpuRanges(numa_node0_cpus, 0);
                    numa_ranges.insert(numa_ranges.end(), node0_ranges.begin(), node0_ranges.end());
                }
                if (!numa_node1_cpus.empty()) {
                    std::vector<NumaNodeRange> node1_ranges = parseNumaCpuRanges(numa_node1_cpus, 1);
                    numa_ranges.insert(numa_ranges.end(), node1_ranges.begin(), node1_ranges.end());
                }
            }
            
            return numa_ranges;
        }
        CoreInfo getCoreNumStruct() {
            CoreInfo info = {1, 1, false, 1};
            
            // get the logical core num
            long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
            if (logical_cores != -1) {
                info.logical_cores = logical_cores;
            }
            
            // get the physical core num
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (cpuinfo.is_open()) {
                std::set<std::pair<int, int>> unique_cores;
                std::string line;
                int physical_id = 0, core_id = 0;
                
                while (std::getline(cpuinfo, line)) {
                    if (line.find("physical id") != std::string::npos) {
                        size_t pos = line.find(':');
                        if (pos != std::string::npos) {
                            physical_id = std::stoi(line.substr(pos + 1));
                        }
                    } else if (line.find("core id") != std::string::npos) {
                        size_t pos = line.find(':');
                        if (pos != std::string::npos) {
                            core_id = std::stoi(line.substr(pos + 1));
                        }
                        unique_cores.insert(std::make_pair(physical_id, core_id));
                    }
                }
                info.physical_cores = unique_cores.size();
                info.is_hyperthreading = (info.physical_cores > 0 && info.logical_cores > info.physical_cores);
            }
            // get the numa node num
            FILE* lscpu_output = popen("lscpu | grep -i 'NUMA node(s)' | awk '{print $NF}'", "r");
            if (lscpu_output) {
                char buffer[16];
                if (fgets(buffer, sizeof(buffer), lscpu_output)) {
                    info.numa_nodes = std::atoi(buffer);
                }
                pclose(lscpu_output);
            }
            
            return info;
        }
        void printNodeInfo(std::vector<NumaNodeRange> numa_ranges) {
            std::cout << "\nVerify output:" << std::endl;
            std::cout << "Node index :(";
            for (size_t i = 0; i < numa_ranges.size(); ++i) {
                std::cout << numa_ranges[i].node_index;
                if (i < numa_ranges.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            
            std::cout << "start index :(";
            for (size_t i = 0; i < numa_ranges.size(); ++i) {
                std::cout << numa_ranges[i].start_cpu;
                if (i < numa_ranges.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            
            std::cout << "end index :(";
            for (size_t i = 0; i < numa_ranges.size(); ++i) {
                std::cout << numa_ranges[i].end_cpu;
                if (i < numa_ranges.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
        }
        void clear_cpu_affinity() {
            static CoreInfo info={1, 1, false, 1};
            info = getCoreNumStruct();
#pragma omp parallel
            {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                
                for (int i = 0; i < info.logical_cores; i++) {
                    CPU_SET(i, &cpuset);
                }
                
                int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                if (ret != 0) {
                    perror("pthread_setaffinity_np");
                }
            }
        }
        void check_bind(int see_num){
            // omp_set_num_threads(num_threads);
            static int printOne = 10;
            if(printOne-->0){
                int* tmpArray = new int[see_num];
                int configChange = 0;
                
#pragma omp parallel
                    {
                        int thread_id = omp_get_thread_num();
                        int core_id = sched_getcpu();
                        tmpArray[thread_id] = core_id; // false sharing but not a big deal
                    }
                std::cout << "tid:  ";
                for(int ii = 0; ii < see_num; ii++) {
                    std::cout << std::setw(3) << ii;
                }
                std::cout << std::endl;

                std::cout << "cpus: ";
                for(int ii = 0; ii < see_num; ii++) {
                    std::cout << std::setw(3) << tmpArray[ii];
                }
                std::cout << std::endl;
            }
        }

        void configOMP(int num_threads, int nodeMode, std::vector<NodeIDType> &bindList){
            // default nodeMode is 2 node
            omp_set_num_threads(num_threads);
            static int* dstCpusList = new int[num_threads]; 

            if (num_threads != 1 && nodeMode != 2){
                static int useOne = 1;
                static CoreInfo info={1, 1, false, 1};
                static std::vector<NumaNodeRange> numaInfo = {};
                if(useOne-->0){
                    info = getCoreNumStruct();
                    std::cout << "Number of logical cores: " << info.logical_cores << std::endl;
                    std::cout << "Number of physical cores: " << info.physical_cores << std::endl;
                    std::cout << "is_hyperthreading: " << (info.is_hyperthreading ? "Yes" : "No") << std::endl;
                    std::cout << "Number of numa node: " << info.numa_nodes << std::endl;
                    numaInfo = getNumaNodeStruct();
                    // printNodeInfo(numaInfo);
                }
                if(num_threads > info.logical_cores) num_threads = info.logical_cores;
                if (info.numa_nodes <= 1) // single node machine nodeMode only support 0 and 1
                {
#pragma omp parallel
                    {
                        int bindMode = 0;
                        if(nodeMode == 0 || nodeMode == 1) bindMode = 1; // close
                        else bindMode = 2; // spread
                        int thread_id = omp_get_thread_num();
                        // ensure not to exceed the system core range
                        int target_core = 0;
                        if (thread_id < 6)
                        {
                            target_core = thread_id * (bindMode);//node Mode 1 for close;2 for spread
                        }
                        else
                        {
                            target_core = thread_id + 6 * (bindMode - 1); // 6 is the performance core number
                        }
                        // set CPU affinity
                        cpu_set_t cpuset;
                        CPU_ZERO(&cpuset);
                        CPU_SET(target_core, &cpuset);

                        // bind the current thread to the specified core
                        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
                        {
                            perror("pthread_setaffinity_np");
                        }
                    }
                }
                else
                { // multi node machine
                    if(useOne==0){
                        // 1 shot get dstCpusList
                        // 00 or 01 static
                        std::fill(dstCpusList, dstCpusList + num_threads, -1);
                        if (nodeMode == 0 || nodeMode == 1)
                        {
                            int invertEnable = 0;
                            if (nodeMode == 1)
                            {
                                invertEnable = 1;
                            }
                            for (int j = 0; j < num_threads; j++)
                            {
                                if(j <= (info.physical_cores - 1)){
                                    dstCpusList[j] = numaInfo[j % 2 + invertEnable * 2].start_cpu + (int)(j / 2);
                                }
                                else{
                                    dstCpusList[j] = numaInfo[j % 2 + 2 - invertEnable * 2].start_cpu + (int)((j-info.physical_cores) / 2);
                                }
                            }

                            if (useOne > -1)
                            {
                                printf("Thread Bind to :");
                                for (int i = 0; i < num_threads; ++i)
                                {
                                    printf("%d ", dstCpusList[i]);
                                }
                                printf("\n");
                            }
                        }
                        else if (nodeMode == 4 || nodeMode == 5){// Sequential Binding
                            int invertEnable = 0;
                            if (nodeMode == 5)
                            {
                                invertEnable = 1;
                            }
                            for (int j = 0; j < num_threads; j++)
                            {
                                if(j <= info.logical_cores - 1){
                                    dstCpusList[j] = j+invertEnable*40;// numaInfo[j % 2 + invertEnable * 2].start_cpu + (int)(j / 2);
                                }
                            }
                        }
                        else
                        { // 03 or etc is dynamic
                            int addFlag = 0;
                            for (int j = 0; j < num_threads; j++)
                            {
                                int temp = 0;
                                if (dstCpusList[j] == -1 && dstCpusList[bindList[j]] == -1)
                                {
                                    dstCpusList[j] = numaInfo[0].start_cpu + addFlag;
                                    dstCpusList[bindList[j]] = numaInfo[1].start_cpu + addFlag;
                                    addFlag += 1;
                                }
                            }

                            int subFlag = 1;
                            if (useOne > -1)
                                printf("Thread Bind 2 :");
                            for (int i = 0; i < num_threads; ++i)
                            { // fix -1
                                if (useOne > -1)
                                    printf("%d ", dstCpusList[i]);
                                if (dstCpusList[i] == -1)
                                {
                                    if (info.is_hyperthreading)
                                    {
                                        if (num_threads < info.physical_cores)
                                        {
                                            dstCpusList[i] = info.physical_cores - subFlag;
                                            subFlag++;
                                        }
                                        else
                                        {
                                            dstCpusList[i] = info.logical_cores - subFlag;
                                            subFlag++;
                                        }
                                    }
                                }
                            }
                            if (useOne > -1)
                                printf("\n");
                        }
                    }
#pragma omp parallel
                    {
                        int thread_id = omp_get_thread_num();
                        int target_core = 0;
                        target_core = dstCpusList[thread_id];
                        // set CPU affinity
                        cpu_set_t cpuset;
                        CPU_ZERO(&cpuset);
                        CPU_SET(target_core, &cpuset);

                        // bind the current thread to the specified core
                        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
                        {
                            perror("[ERROR]pthread_setaffinity_np");
                        }
                    }
                }
            }
            static int printOne = 1;
            if(printOne-->0){
                int* tmpArray = new int[num_threads];
                int configChange = 0;
                
#pragma omp parallel
                    {
                        int thread_id = omp_get_thread_num();
                        int core_id = sched_getcpu();
                        tmpArray[thread_id] = core_id; // false sharing but not a big deal
                    }
                std::cout << "tid:  ";
                for(int ii = 0; ii < num_threads; ii++) {
                    std::cout << std::setw(3) << ii;
                }
                std::cout << std::endl;

                std::cout << "cpus: ";
                for(int ii = 0; ii < num_threads; ii++) {
                    std::cout << std::setw(3) << tmpArray[ii];
                }
                std::cout << std::endl;
            }
        }
    };

template<typename T>
inline py::array vec2npy(const std::vector<T> &vec)
{
    // need to let python garbage collector handle C++ vector memory 
    // see https://github.com/pybind/pybind11/issues/1042
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void *v)
                               { delete reinterpret_cast<std::vector<T> *>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

PYBIND11_MODULE(sampler_core, m)
{
    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<std::vector<NodeIDType> &, std::vector<NodeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<TimeStampType> &,
                      std::vector<TimeStampType> &, std::vector<NodeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<EdgeIDType> &,std::vector<EdgeIDType> &,
                      NodeIDType, NodeIDType>())
        .def("row", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.row); })
        .def("col", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.col); })
        .def("eid", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.eid); })
        .def("ts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.ts); })
        .def("dts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.dts); })
        .def("nodes", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.nodes); })
        .def("uniE", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.uniE); })
        .def("invE", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.invE); })
        .def("cntE", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.cntE); })
        .def("dim_in", [](const TemporalGraphBlock &tgb) { return tgb.dim_in; })
        .def("dim_out", [](const TemporalGraphBlock &tgb) { return tgb.dim_out; })
        .def("tot_time", [](const TemporalGraphBlock &tgb) { return tgb.tot_time; })
        .def("ptr_time", [](const TemporalGraphBlock &tgb) { return tgb.ptr_time; })
        .def("search_time", [](const TemporalGraphBlock &tgb) { return tgb.search_time; })
        .def("sample_time", [](const TemporalGraphBlock &tgb) { return tgb.sample_time; })
        .def("coo_time", [](const TemporalGraphBlock &tgb) { return tgb.coo_time; });
    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<std::vector<EdgeIDType> &, std::vector<EdgeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<TimeStampType> &,
                      int, int, int, std::vector<int> &, bool, bool,
                      int, TimeStampType>())
        .def("sample", &ParallelSampler::sample)
        .def("reset", &ParallelSampler::reset)
        .def("get_ret", [](const ParallelSampler &ps) { return ps.ret; })
        .def("configOMP", &ParallelSampler::configOMP)
        .def("clear_cpu_affinity", &ParallelSampler::clear_cpu_affinity)
        .def("check_bind", &ParallelSampler::check_bind, py::arg("see_num"));
}
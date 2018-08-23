#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <assert.h>

using namespace std;

void dispVector( const vector<double>& v , char separator);

double vec_mean( const vector<double>& v );

double vec_standard_deviation( const vector<double>& v, double mean );

void check_normalization( const vector<double>& v );

double vec_normalize( vector<double>& v );

size_t count_lines( const string &filename );

void disp_column_check_normalization( const vector<double>& v, int col, int dim, size_t n);

void read_and_normalize_data(const string &filename, int dim, int start, int end, std::vector<double> &data_normalized);

void read_labels(const string &filename, int dim, int start, int end, std::vector<double> &data);

void data_read_normalize_from_file(const string &filename, int dim, size_t training, size_t validation, size_t testing,
	vector<double> &data_training, vector<double> &data_validation, vector<double> &data_testing );

void labels_read_from_file(const string &filename, int dim, size_t training, size_t validation, size_t testing,
	vector<double> &labels_training, vector<double> &labels_validation, vector<double> &labels_testing );
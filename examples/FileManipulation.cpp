#include "FileManipulation.h"

using namespace std;

void dispVector( const vector<double>& v , char separator){
	for ( auto entry : v)
		cout << entry << separator;
	cout << endl;
}

double vec_mean( const vector<double>& v ){
	double sum = std::accumulate(v.begin(), v.end(), 0.0);
	double mean = sum / v.size();
	return mean;
}

double vec_standard_deviation( const vector<double>& v, double mean ){
	vector<double> diff = v;
	// Computing v-mean(v)
	std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0 /*Initial value for the accumulator*/);
	double stdev = std::sqrt(sq_sum / (diff.size()-1.0));
	return stdev;
}

void check_normalization( const vector<double>& v ){
	cout << "mean = "  << vec_mean( v ) << endl;
	cout << "stdev = "  << vec_standard_deviation( v, vec_mean( v ) ) << endl;
}

double vec_normalize( vector<double>& v ){
	double mean = 0;
	double stdev = 0;
	mean = vec_mean( v );
	stdev = vec_standard_deviation( v, mean );

	// Computing v = v-mean(v)
	std::transform(v.begin(), v.end(), v.begin(), [mean](double x) { return x - mean; });
	// Computing v = v/std(v)
	std::transform(v.begin(), v.end(), v.begin(), [stdev](double x) { return x/stdev; });
}

size_t count_lines( const string &filename ){
	size_t count_lines = 0;
	ifstream file( filename );
	count_lines = std::count( std::istreambuf_iterator<char>( file ), std::istreambuf_iterator<char>(), '\n' );
	file.close();
	// cout << "# File " << filename << " has " << count_lines << " lines" << endl;
	return count_lines;
}

void disp_column_check_normalization( const vector<double>& v, int col, int dim, size_t n){
	size_t cntCols = 0;
	size_t cntLines = 0;

	std::vector<double> col_vec;
	for (cntLines = 0; cntLines < n; ++cntLines)
		col_vec.push_back(v[cntLines*dim+col]);
	
	dispVector(col_vec, '\n');
	check_normalization(col_vec);
	cout << endl;
}

void read_and_normalize_data(const string &filename, int dim, int start, int end, std::vector<double> &data_normalized) {
	
	std::vector<double> data[dim];
	size_t size = end - start;
	// cout << "start " << start << endl;
	// cout << "end   " << end   << endl;
	// cout << "size  " << size  << endl;

	if ( size > 0 ) {

		ifstream file(filename);
		string line;
		size_t cntCols = 0;
		size_t cntLines = 0;

		// Read from file, and separate into 'dim' vectors
		while ( cntLines < end ) {
			getline(file, line);
			if ( cntLines >= start){
				// cout << "reading " << cntLines << endl;
				// cout << cntLines << " : " << line << endl;
				istringstream sl(line);
				string s;
				cntCols = 0;
				while (getline(sl, s, ',')){
					data[cntCols].push_back(stod(s));
					cntCols++;
				}
			}
			cntLines++;
		}
		file.close();

		// Normalize each column
		for (cntCols = 0; cntCols < dim; ++cntCols) {
			vec_normalize( data[cntCols] );
			// check_normalization( data[cntCols] );
		}

		// Merge columns into one vector
		for (cntLines = 0; cntLines < size; ++cntLines)
			for (cntCols = 0; cntCols < dim; ++cntCols)
				data_normalized.push_back( data[cntCols][cntLines] );
	}
}

void read_labels(const string &filename, int dim, int start, int end, std::vector<double> &data) {
	
	size_t size = end - start;
	// cout << "start " << start << endl;
	// cout << "end   " << end   << endl;
	// cout << "size  " << size  << endl;

	if ( size > 0 ) {

		ifstream file(filename);
		string line;
		size_t cntLines = 0;

		// Read from file
		while ( cntLines < end ) {
			getline(file, line);
			if ( cntLines >= start){
				// cout << "cntLines: " << cntLines << endl;
				// cout << "line: " << line << endl;
				istringstream sl(line);
				string s;
				while (getline(sl, s, ',')){
					data.push_back(stod(s));
				}
			}
			cntLines++;
		}
		file.close();
	}
}

void data_read_normalize_from_file(const string &filename, int dim, size_t training, size_t validation, size_t testing,
	vector<double> &data_training, vector<double> &data_validation, vector<double> &data_testing ) {
	assert( (training+validation+testing) <= count_lines( filename ) && "Not enough input data in data file");
	read_and_normalize_data(filename, dim, 0, 									training, 											 data_training);
	read_and_normalize_data(filename, dim, training, 						training + testing, 						 data_testing);
	read_and_normalize_data(filename, dim, training + testing,	training + testing + validation, data_validation);
}

void labels_read_from_file(const string &filename, int dim, size_t training, size_t validation, size_t testing,
	vector<double> &labels_training, vector<double> &labels_validation, vector<double> &labels_testing ) {
	assert( (training+validation+testing) <= count_lines( filename ) && "Not enough input data in labels file");
	read_labels(filename, dim, 0, 									training, 											 labels_training);
	read_labels(filename, dim, training, 						training + testing, 						 labels_testing);
	read_labels(filename, dim, training + testing,	training + testing + validation, labels_validation);
}

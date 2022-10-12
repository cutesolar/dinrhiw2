
#ifndef argparser_tab_h
#define argparser_tab_h

#include <string>
#include <vector>

void parse_commandline(int argc, char** argv,
		       std::string& datafilename,
		       std::string& nnfilename,
		       std::string& lmethod, 
		       std::vector<std::string>& lmods,
		       std::vector<unsigned int>& arch,
		       unsigned int& cmdmode,
		       unsigned int& secs,
		       unsigned int& samples,
		       unsigned int& threads,
		       unsigned int& dataSize,
		       unsigned int& SIMULATION_DEPTH,
		       bool& no_init,
		       bool& load, 
		       bool& overfit, 
		       bool& adaptive,
		       bool& negfeedback,
		       unsigned int& deep,
		       bool& residual,
		       bool& dropout,
		       bool& crossvalidation,
		       bool& batchnorm,
		       bool& help, 
		       bool& verbose);

#endif

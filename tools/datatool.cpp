/*
 * tool for manipulating whiteice::dataset files.
 * 
 * TODO add commands/extend commands to handle
 * individual data vectors 
 * (remove, move, copy, swap vectors)
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#include <dinrhiw/dinrhiw.h>
#include <dinrhiw.h>

#include <exception>

#include <string>
#include <string.h>
#include <vector>

using namespace whiteice;


bool parse(int argc, char** argv, 
	   std::string& action, 
	   std::vector<std::string>& options, 
	   std::string& datafile1, 
	   std::string& datafile2);

void print_usage();


bool importdata(const std::string& filename, 
		unsigned int cluster,
		whiteice::dataset< math::blas_real<float> >* data);

bool exportdata(const std::string& filename,
		unsigned int cluster,
		whiteice::dataset< math::blas_real<float> >* data);



int main(int argc, char** argv)
{
  try{
    std::string action;
    std::vector<std::string> options;
    std::string datafile1;
    std::string datafile2;
    
    if(!parse(argc, argv, action, options, datafile1, datafile2)){
      print_usage();
      return -1;
    }
    
    // implementation of commands
    
    whiteice::dataset< math::blas_real<float> >* data =
      new whiteice::dataset< math::blas_real<float> >();
    

    if(action != "create" || options.size() != 0){
      if(data->load(datafile1) == false){
	std::cout << "Cannot access file: " << datafile1 << std::endl;
	delete data;
	return -1;
      }
    }
    
    
    if(action == "list"){
      std::cout << "Number of clusters: " 
		<< data->getNumberOfClusters() << std::endl;
      
      for(unsigned int i=0;i<data->getNumberOfClusters();i++){
	std::cout << "Cluster " << i << " \"" << data->getName(i) << "\" : "
		  << data->dimension(i) << " dimensions, "
		  << data->size(i) << " points." << std::endl;
	
	std::vector<whiteice::dataset< math::blas_real<float> >::data_normalization> norms;
	if(data->getPreprocessings(i, norms)){
	  std::cout << "Cluster " << i << " \"" << data->getName(i) << "\" : ";
	  
	  for(unsigned int j=0;j<norms.size();j++){
	    if(j != 0) std::cout << " , ";
	    if(norms[j] == whiteice::dataset< math::blas_real<float> >::dnMeanVarianceNormalization)
	      std::cout << "meanvar";
	    else if(norms[j] == whiteice::dataset< math::blas_real<float> >::dnSoftMax)
	      std::cout << "outlier";
	    else if(norms[j] == whiteice::dataset< math::blas_real<float> >::dnCorrelationRemoval)
	      std::cout << "pca";
	    else if(norms[j] == whiteice::dataset< math::blas_real<float> >::dnLinearICA)
	      std::cout << "ica";
	    else
	      std::cout << "<unknown>" << std::endl;
	  }
	}
	
	std::cout << std::endl;
      }
      
      
      delete data;
      return 0;
    }
    else if(action == "print" && options.size() > 0){
      
      unsigned int c, begIndex, endIndex;
      
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      c = (unsigned int)strtol(startp, &endp, 10);
      
      if(!(*endp == '\0' && endp != startp)){
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
      if(data->getNumberOfClusters() <= c){
	std::cout << "No such cluster." << std::endl;
	delete data;
	return -1;
      }
      
      begIndex = 0;
      if(data->size(c) > 0)
	endIndex = data->size(c) - 1;
      else
	endIndex = 0;
      
      
      if(options.size() >= 2){
	startp = options[1].c_str();
	endp = (char*)startp;
	begIndex = (unsigned int)strtol(startp, &endp, 10);
	
	if(!(*endp == '\0' && endp != startp)){
	  std::cout << "Syntax error." << std::endl;
	  delete data;
	  return -1;
	}
	
	if(options.size() == 3){
	  startp = options[2].c_str();
	  endp = (char*)startp;
	  endIndex = (unsigned int)strtol(startp, &endp, 10);
	  
	  if(!(*endp == '\0' && endp != startp)){
	    std::cout << "Syntax error." << std::endl;
	    delete data;
	    return -1;
	  }
	}
	else if(options.size() > 3){
	  std::cout << "Syntax error." << std::endl;
	  delete data;
	  return -1;
	}
      }
      
      if(data->size(c) > 0){
	
	std::cout << "Cluster " << c << " \"" << data->getName(c) << "\" points: " 
		  << begIndex << " - " << endIndex
		  << std::endl;
	endIndex++;
	
	for(unsigned int i=begIndex;i<endIndex;i++){
	  math::vertex<> xx = data->access(c, i);
	  if(data->invpreprocess(c, xx))
	    std::cout << i << ": " << xx << std::endl;
	  else
	    std::cout << i << ": preprocessing failed." << std::endl;
	}
	
      }
      else{
	std::cout << "Cluster " << c << " \"" << data->getName(c) << "\" is empty."
		  << std::endl;
      }
      
      
      delete data;
      return 0;
    }
    else if(action == "create" && options.size() == 0){
      
      // ok (save at the end saves new empty dataset)
    }
    else if(action == "create" && (options.size() == 1 || options.size() == 2)){
      // tries to add a new cluster
      
      const char *startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int dim = (unsigned int)strtol(startp, &endp, 10);
      std::string name = "N/A";
      if(options.size() == 2)
	name = options[1];
      
      if(*endp == '\0' && endp != startp){
	std::cout << action << " " << options[0] << " " << options[1] << std::endl;
	
	if(data->createCluster(name, dim) == false){
	  std::cout << "Couldn't create cluster" << std::endl;
	  delete data;
	  return -1;
	}
	
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
    }
    else if(action == "import"){
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int cluster = (unsigned int)strtol(startp, &endp, 10);
      
      if(*endp == '\0' && endp != startp){
	if(importdata(datafile2, cluster, data) == false){
	  std::cout << "Importing failed. No such cluster or badly formated ascii file."
		    << std::endl;
	  delete data;
	  return -1;
	}
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;	
      }
    }
    else if(action == "export"){
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int cluster = (unsigned int)strtol(startp, &endp, 10);
      
      if(*endp == '\0' && endp != startp){
	if(exportdata(datafile2, cluster, data) == false){
	  std::cout << "Exporting failed. No such cluster or filesystem I/O error."
		    << std::endl;
	  delete data;
	  return -1;	
	}
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
    }
    else if(action == "add" && options.size() == 2){
      whiteice::dataset< math::blas_real<float> >* data2 =
	new whiteice::dataset< math::blas_real<float> >();
      
      if(data2->load(datafile2) == false){
	std::cout << "Cannot load file: " << datafile2 << std::endl;
	delete data;
	return -1;
      }
      
      
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int c1 = (unsigned int)strtol(startp, &endp, 10);
      
      if(!(*endp == '\0' && endp != startp)){
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
      
      startp = options[1].c_str();
      endp = (char*)startp;
      unsigned int c2 = (unsigned int)strtol(startp, &endp, 10);
      
      if(!(*endp == '\0' && endp != startp)){
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
      
      if(c1 >= data->getNumberOfClusters() ||
	 c2 >= data2->getNumberOfClusters()){
	std::cout << "No such cluster(s)." << std::endl;
	delete data;
	return -1;
      }
      
      
      if(data->dimension(c1) != data2->dimension(c2)){
	std::cout << "Clusters' data dimenions must be same" 
		  << std::endl;
	delete data;
	return -1;
      }
      
      
      for(unsigned int i=0;i<data2->size(c2);i++){
	if(data->add(c1, data2->access(c2, i)) == false){
	  std::cout << "Internal error in (add(), access() methods)" 
		    << std::endl;
	  
	  delete data;
	  return -1;
	}
      }
      
      
      delete data2;
      
    }
    else if((action == "move" || action == "copy") && options.size() == 2){
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int c1 = (unsigned int)strtol(startp, &endp, 10);
      
      if(!(*endp == '\0' && endp != startp)){
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
      
      startp = options[1].c_str();
      endp = (char*)startp;
      unsigned int c2 = (unsigned int)strtol(startp, &endp, 10);
      
      if(!(*endp == '\0' && endp != startp)){
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
      
      if(c1 >= data->getNumberOfClusters() ||
	 c2 >= data->getNumberOfClusters()){
	std::cout << "No such cluster(s)." << std::endl;
	delete data;
	return -1;
      }
      
      
      if(data->dimension(c1) != data->dimension(c2)){
	std::cout << "Clusters' data dimenions must be same" 
		  << std::endl;
	delete data;
	return -1;
      }
      
      
      for(unsigned int i=0;i<data->size(c2);i++){
	if(data->add(c1, data->access(c2, i)) == false){
	  std::cout << "Internal error in (add(), access() methods)" 
		    << std::endl;
	  
	  delete data;
	  return -1;
	}
      }
      
      
      if(action == "move")
	data->clearAll(c2);
      
    }
    else if(action == "clear" && options.size() == 1){
      
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int cluster = (unsigned int)strtol(startp, &endp, 10);
      
      if(*endp == '\0' && endp != startp){
	if(data->clearData(cluster) == false){ // do NOT remove preprocessing information
	  if(cluster >= data->getNumberOfClusters()){
	    std::cout << "Cannot clear cluster. Cluster doesn't exist." 
		      << std::endl;
	    delete data;
	    return -1;
	  }
	  else{
	    std::cout << "Cannot clear cluster " 
		      << cluster << std::endl;
	    delete data;
	    return -1;
	  }
	}
	
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
    }
    else if(action == "remove" && options.size() == 1){
      
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int cluster = (unsigned int)strtol(startp, &endp, 10);
      
      if(*endp == '\0' && endp != startp){
	if(data->removeCluster(cluster) == false){
	  if(cluster >= data->getNumberOfClusters()){
	    std::cout << "Cannot remove cluster. Cluster doesn't exist." 
		      << std::endl;
	    delete data;
	    return -1;
	  }
	  else{
	    std::cout << "Cannot remove cluster " 
		      << cluster << std::endl;
	    delete data;
	    return -1;
	  }
	}
	
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;
      }
      
    }
    else if(action == "data" && options.size() == 1){
      // resamples data down to N datapoints

      // loads gets
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      const unsigned int number = (unsigned int)strtol(startp, &endp, 10);
      
      if(data->downsampleAll(number) == false){
	std::cout << "Downsampling data to " << number
		  << " datapoint(s) failed." << std::endl;
	delete data;
	return -1;
      }
      
      
    }
    else if(action == "padd" && options.size() > 1){
      // adds preprocessing to cluster (names: meanvar, outlier, pca)
      
      // loads gets
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int cluster = (unsigned int)strtol(startp, &endp, 10);
      
      if(*endp == '\0' && endp != startp){
	
	if(cluster >= data->getNumberOfClusters()){
	  std::cout << "No such cluster." << std::endl;
	  delete data;
	  return -1;
	}
	
	
	std::vector<dataset< math::blas_real<float> >::data_normalization> norms;
	
	for(unsigned int i=1;i<options.size();i++){
	  if(options[i] == "meanvar"){
	    norms.push_back(dataset< math::blas_real<float> >::dnMeanVarianceNormalization);
	  }
	  else if(options[i] == "outlier"){
	    norms.push_back(dataset< math::blas_real<float> >::dnSoftMax);
	    
	  }
	  else if(options[i] == "pca"){
	    norms.push_back(dataset< math::blas_real<float> >::dnCorrelationRemoval);
	  }
	  else if(options[i] == "ica"){
	    norms.push_back(dataset< math::blas_real<float> >::dnLinearICA);
	  }
	  else{
	    std::cout << "Unrecognized preprocessing option(s)."
		      << std::endl;
	    delete data;
	    return -1;
	  }
	}
	
	
	for(unsigned int i=0;i<norms.size();i++)
	  if(data->preprocess(cluster, norms[i]) == false){
	    std::cout << "Preprocessing with ";
	    if(i == 0) std::cout << "1st ";
	    else if(i == 1) std::cout << "2nd ";
	    else if(i == 2) std::cout << "3rd ";
	    else std::cout << (i+1) << "th ";
	    std::cout << "method failed." << std::endl;
	    
	    delete data;
	    
	    return -1;
	  }
	
	
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;	
      }
      
    }
    else if(action == "premove" && options.size() > 1){
      // removes preprocessing from cluster (names: meanvar, outlier, pca)
      
      // loads gets
      const char* startp = options[0].c_str();
      char *endp = (char*)startp;
      unsigned int cluster = (unsigned int)strtol(startp, &endp, 10);
      
      if(*endp == '\0' && endp != startp){
	
	if(cluster >= data->getNumberOfClusters()){
	  std::cout << "No such cluster." << std::endl;
	  delete data;
	  return -1;
	}
	
	
	std::vector<dataset< math::blas_real<float> >::data_normalization> norms;
	
	for(unsigned int i=1;i<options.size();i++){
	  if(options[i] == "meanvar"){
	    norms.push_back(dataset< math::blas_real<float> >::dnMeanVarianceNormalization);
	  }
	  else if(options[i] == "outlier"){
	    norms.push_back(dataset< math::blas_real<float> >::dnSoftMax);
	    
	  }
	  else if(options[i] == "pca"){
	    norms.push_back(dataset< math::blas_real<float> >::dnCorrelationRemoval);
	  }
	  else if(options[i] == "ica"){
	    norms.push_back(dataset< math::blas_real<float> >::dnLinearICA);
	  }
	  else{
	    std::cout << "Unrecognized preprocessing options"
		      << std::endl;
	    delete data;
	    return -1;
	  }
	}
	
	// LIST = CURRENT \ NORMS = CURRENT AND (CONJ(NORMS))
	std::vector<dataset< math::blas_real<float> >::data_normalization> current, list;
	if(data->getPreprocessings(cluster, current) == false){
	  std::cout << "No such cluster or internal error." << std::endl;
	  delete data;
	  return -1;
	}
	
	for(unsigned int i=0;i<current.size();i++){
	  bool removed = false;
	  for(unsigned int j=0;j<norms.size();j++)
	    if(norms[j] == current[i])
	      removed = true;
	  
	  
	  if(!removed)
	    list.push_back(current[i]);
	}
	
	
	if(data->convert(cluster, list) == false){
	  std::cout << "Changing preprocessings failed" << std::endl;
	  delete data;
	  return -1;
	}
	
	
      }
      else{
	std::cout << "Syntax error." << std::endl;
	delete data;
	return -1;	
      }
      
    }
    else{
      std::cout << "Internal error: unrecognized command line parameter" 
		<< std::endl;
      delete data;
      return -1;
    }
    
    
    
    if(data->save(datafile1) == false){
      std::cout << "Couldn't save file: " << datafile1 << std::endl;
      delete data;
      return -1;
    }

    
    
    delete data;
    return 0;
  }
  catch(std::exception& e){
    std::cout << "Fatal error: unexpected exception. Reason: " 
	      << e.what() << std::endl;
    return -1;
  }
}



bool parse(int argc, char** argv, 
	   std::string& action, 
	   std::vector<std::string>& options, 
	   std::string& datafile1, 
	   std::string& datafile2)
{
  if(argc < 3) return false;
  
  // parses command
  
  if(argv[1][0] != '-') return false;
  char *start = argv[1] + 1;
  char *ptr = start;
  
  
  while(*ptr != ':' && *ptr != '\0') 
    ptr++;
  
  if(*ptr == '\0'){
    action = start;
  }
  else{
    *ptr = '\0';
    action = start;

    ptr++;
    start = ptr;
    
    // gets options
    // separated with ':'
    while(1){
      while(*ptr != ':' && *ptr != '\0')
	ptr++;
      
      if(*ptr == '\0'){
	options.push_back(start);
	break; // end of options
      }
      else{
	*ptr = '\0';
	ptr++;
	
	options.push_back(start);
	start = ptr;
      }
    }
  }
  
  
  if(action == "add" || action == "import" || action == "export"){
    if(argc != 4)
      return false;
    
    datafile1 = argv[2];
    datafile2 = argv[3];
  }
  else{
    if(argc != 3)
      return false;
    
    datafile1 = argv[2];
  }
  
  
  // checks action is recognized
  
  std::vector<std::string> okcmds;
  okcmds.push_back("list");
  okcmds.push_back("print");
  okcmds.push_back("create");
  okcmds.push_back("add");
  okcmds.push_back("import");
  okcmds.push_back("export");
  okcmds.push_back("move");
  okcmds.push_back("copy");
  okcmds.push_back("clear");
  okcmds.push_back("remove");
  okcmds.push_back("padd");
  okcmds.push_back("premove");
  okcmds.push_back("data");
  
  
  for(unsigned int i=0;i<okcmds.size();i++)
    if(okcmds[i] == action) return true;
  
  return false; // unrecognized command
}



void print_usage()
{
  printf("Usage: datatool <command> <datafile> [asciifile | datafile]\n");
  printf("A tool for manipulating whiteice::dataset files.\n");
  printf("\n");
  printf(" -list                      lists clusters, number of datapoints, preprocessings.\n");
  printf("                            (default action)\n");
  printf(" -print[:<c1>[:<b>[:<e>]]]  prints contents of cluster c1 (indexes [<b>,<e>])\n");
  printf(" -create                    creates new empty dataset (<dataset> file doesn't exist)\n");
  printf(" -create:<dim>[:name]       creates new empty <dim> dimensional dataset cluster\n");
  printf(" -import:<c1>               imports data from comma separated CSV ascii file to cluster c1\n");
  printf(" -export:<c1>               exports data from cluster c1 to comma separated CSV ascii file\n");
  printf(" -add:<c1>:<c2>             adds data from another datafile cluster c2 to c1\n");
  printf(" -move:<c1>:<c2>            moves data (internally) from cluster c2 to c1\n");
  printf(" -copy:<c1>:<c2>            copies data (internally) from cluster c2 to c1\n");
  printf(" -clear:<c1>                clears dataset cluster c1 (but doesn't remove cluster)\n");
  printf(" -remove:<c1>               removes dataset cluster c1\n");
  printf(" -padd:<c1>:<name>+         adds preprocessing(s) to cluster\n");
  printf(" -premove:<c1>:<name>+      removes preprocesing(s) from cluster\n");
  printf("                            preprocess names: meanvar, outlier, pca, ica\n");
  printf("                            note: ica implementation is unstable and may not work\n");
  printf(" -data:N                    jointly resamples all cluster sizes down to N datapoints\n");
  printf("\n");
  printf("This program is distributed under GPL license <tomas.ukkonen@iki.fi> (other licenses available).\n");
} 


bool importdata(const std::string& filename, 
		unsigned int cluster,
		whiteice::dataset< math::blas_real<float> >* data)
{
  if(data == 0) return false;
  
  if(data->getNumberOfClusters() <= cluster)
    return false;
  
  FILE* fp = fopen(filename.c_str(), "rt");
  if(fp == 0 || ferror(fp)){
    if(fp) fclose(fp);
    return false;
  }
  
  // import format is
  // <file> = (<line>"\n")*
  // <line> = <vector> = "%f %f %f %f ... "
  // 
  
  while(!feof(fp)){
    
    // reads a single line
    unsigned int len = 4096, used = 0;
    char* buffer = (char*)malloc(len);
    char* s = buffer;
    
    if(buffer == 0){ 
      fclose(fp); 
      return false;
    }
    
    
    while(1){
      if(fgets(s, len - used, fp) == 0){
	if(feof(fp)){
	  *s = '\0';
	  break;
	}
	else{
	  free(buffer);
	  fclose(fp); 
	  return false;
	}
      }
      
      unsigned int n = strlen(s);
      if(s[n-1] == '\n') // new line found
	break;
      
      // need more space
      used += n;
      
      char* p = (char*)realloc(buffer, len + 100);
      if(p == 0){
	free(buffer);
	fclose(fp); 
	return false;
      }
      
      buffer = p;
      s = buffer + used;
      len += 100;
    }
    
    
    // intepretes buffer as a vector
    math::vertex< math::blas_real<float> > line;
    s = buffer;
    unsigned int index = 0;
    
    while(*s != '\n' && *s != '\0'){
      char* prev = s;
      float v = strtof(s, &s);
      if(s == prev) // no progress
	break;
      
      line.resize(index+1);
      line[index] = v;
      index++;
      
      while(*s == ' ' || *s == ',') s++;
    }

    // std::cout << line << std::endl;
    
    if(line.size() != data->dimension(cluster) && index > 0){
      free(buffer);
      fclose(fp); 
      return false;
    }
    else if(index == data->dimension(cluster)){
      data->add(cluster, line);
    }
    
    free(buffer);
    buffer = NULL;
  }
  
  
  fclose(fp);
  
  return true;
}



bool exportdata(const std::string& filename,
		unsigned int cluster,
		whiteice::dataset< math::blas_real<float> >* data)
{
  if(data == 0) 
    return false;
  
  if(data->getNumberOfClusters() <= cluster)
    return false;
  
  FILE* fp = fopen(filename.c_str(), "wt");
  if(fp == 0 || ferror(fp)){
    if(fp) fclose(fp);
    return false;
  }
  
  // export format is
  // <file> = (<line>"\n")*
  // <line> = <vector> = "%f %f %f %f ... "
  // 
  
  const unsigned int N = data->size(cluster);
  
  for(unsigned int i=0;i<N;i++){
    whiteice::math::vertex< math::blas_real<float> > v;
    v = data->access(cluster, i);
    data->invpreprocess(cluster, v);
    
    if(v.size() > 0){
      fprintf(fp, "%f", v[0].real());
    }
    
    for(unsigned int j=1;j<v.size();j++){
      fprintf(fp, " %f", v[j].real());
    }
    
    fprintf(fp, "\n");
  }
  
  
  fclose(fp);
  
  return true;
}




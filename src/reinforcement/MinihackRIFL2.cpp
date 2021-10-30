
#include "MinihackRIFL2.h"


namespace whiteice
{
  // observation space size is 107 (9x9 char environment plus player stats)  and action is one-hot-encoded value
  template <typename T>
  MinihackRIFL2<T>::MinihackRIFL2(const std::string& pythonScript) : RIFL_abstract2<T>(8, 107)
  {
    Py_Initialize();
    
    PyRun_SimpleString("x = 0"); // dummy (needed?)

    pythonFile = fopen(pythonScript.c_str(), "r");

    if(pythonFile == NULL) assert(0); // FIXME proper error handling

    filename = pythonScript;
    PyRun_SimpleFile(pythonFile, filename.c_str());

    main_module = PyImport_AddModule("__main__");
    global_dict = PyModule_GetDict(main_module);

    if(main_module == NULL || global_dict == NULL){
      assert(0); // FIXME proper error handling
    }

    getStateFunc = PyDict_GetItemString(global_dict, (const char*)"minihack_getState");
    performActionFunc = PyDict_GetItemString(global_dict, (const char*)"minihack_performAction");

    if(getStateFunc == NULL || performActionFunc == NULL){
      assert(0); // FIXME proper error handling
    }
    
  }

  template <typename T>
  MinihackRIFL2<T>::~MinihackRIFL2()
  {
    Py_DECREF(getStateFunc);
    Py_DECREF(performActionFunc);
    Py_DECREF(global_dict);
    Py_DECREF(main_module);
    
    if(pythonFile) fclose(pythonFile);    
  }

  template <typename T>
  bool MinihackRIFL2<T>::getState(whiteice::math::vertex<T>& state)
  {
    PyObject *result = NULL;
    
    result = PyObject_CallFunction(getStateFunc, NULL);

    if(result == NULL) return false;

    if(PyList_CheckExact(result) != 1){
      Py_DECREF(result);
      return false;
    }

    const unsigned long SIZE = (unsigned long)PyList_Size(result);

    if(SIZE > 0){

      if(state.resize(SIZE) == false){
	Py_DECREF(result);
	return false;
      }

      for(unsigned long index=0;index<SIZE;index++){
	PyObject* item = PyList_GetItem(result, (Py_ssize_t)index);

	if(PyLong_CheckExact(item) != 1){
	  Py_DECREF(item);
	  Py_DECREF(result);

	  return false;
	}

	state[index] = T(PyLong_AsDouble(item));		

	Py_DECREF(item);
      }
    }

    Py_DECREF(result);

    return true;
  }

  template <typename T>
  bool MinihackRIFL2<T>::performAction(const whiteice::math::vertex<T>& action,
				       whiteice::math::vertex<T>& newstate,
				       T& reinforcement, bool& endFlag)
  {
    // [state, reward, done] = minihack_performAction(action) (action is integer 0..7)

    if(action.size() <= 0) return false;

    // maps one-hot-encoded probabilistic action to integer action 0-7 (8 values)
    unsigned long ACTION = 0;

    {
      const T temperature = T(1.0f);
      T psum = T(0.0f);
      std::vector<T> p;

      for(unsigned int i=0;i<action.size();i++){
	auto value = action[i];

	if(value < T(-6.0f)) value = T(-6.0f);
	else if(value > T(+6.0f)) value = T(+6.0f);

	psum += exp(value/temperature);
	p.push_back(psum);
      }

      for(unsigned int i=0;i<p.size();i++)
	p[i] /= psum;

      psum = T(0.0f);
      for(unsigned int i=0;i<p.size();i++){
	p[i] += psum;
	psum += p[i];
      }

      T r = rng.uniform();
      
      unsigned long index = 0;

      while(p[index] < r){
	index++;
	if(index >= p.size()){
	  index = p.size()-1;
	  break;
	}
      }
      
      ACTION = index;
    }

    
    PyObject *result = NULL;
    
    result = PyObject_CallFunction(performActionFunc, "k", (unsigned long)ACTION);

    if(result == NULL) return false;
    
    // [state, reward, done] = minihack_performAction(action)
    // there are now multiple return values state (int list to double list), reward is float, done is boolean flag

    
    // check return value is a list with 3 elements

    if(PyList_CheckExact(result) != 1){
      Py_DECREF(result);
      return false;
    }

    if(PyList_Size(result) != 3){
      Py_DECREF(result);
      return false;
    }

    // extract return values from the list

    PyObject* stateObj = PyList_GetItem(result, 0);
    PyObject* rewardObj = PyList_GetItem(result, 1);
    PyObject* doneObj = PyList_GetItem(result, 2);

    if(stateObj == NULL || rewardObj == NULL || doneObj == NULL){

      if(stateObj) Py_DECREF(stateObj);
      if(rewardObj) Py_DECREF(rewardObj);
      if(doneObj) Py_DECREF(doneObj);
      
      Py_DECREF(result);
      return false;
    }


    if(PyList_CheckExact(stateObj) != 1){
      Py_DECREF(stateObj);
      Py_DECREF(rewardObj);
      Py_DECREF(doneObj);
      
      Py_DECREF(result);
      return false;
    }

    const unsigned long SIZE = (unsigned long)PyList_Size(stateObj);

    if(SIZE > 0){

      if(newstate.resize(SIZE) == false){
	Py_DECREF(stateObj);
	Py_DECREF(rewardObj);
	Py_DECREF(doneObj);
	
	Py_DECREF(result);
	return false;
      }

      for(unsigned long index=0;index<SIZE;index++){
	PyObject* item = PyList_GetItem(stateObj, (Py_ssize_t)index);

	if(PyLong_CheckExact(item) != 1){
	  Py_DECREF(item);

	  Py_DECREF(stateObj);
	  Py_DECREF(rewardObj);
	  Py_DECREF(doneObj);
	  
	  Py_DECREF(result);

	  return false;
	}
	
	newstate[index] = T(PyLong_AsDouble(item));
	
	Py_DECREF(item);
      }
    }

    if(PyFloat_Check(rewardObj) != 1){
      Py_DECREF(stateObj);
      Py_DECREF(rewardObj);
      Py_DECREF(doneObj);
    
      Py_DECREF(result);
      
      return false;
    }

    reinforcement = T(PyFloat_AsDouble(rewardObj));
    
    if(PyBool_Check(doneObj) != 1){
      Py_DECREF(stateObj);
      Py_DECREF(rewardObj);
      Py_DECREF(doneObj);
    
      Py_DECREF(result);
      
      return false;
    }

    if(doneObj == Py_False)
      endFlag = false;
    else
      endFlag = true;
    
    Py_DECREF(stateObj);
    Py_DECREF(rewardObj);
    Py_DECREF(doneObj);
    
    Py_DECREF(result);    
    
    return true;
  }
  
  

  template class MinihackRIFL2< math::blas_real<float> >;
  template class MinihackRIFL2< math::blas_real<double> >;  
  
};

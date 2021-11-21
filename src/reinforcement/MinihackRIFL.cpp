/*
 * NOTE: It is important that we don't call Py_DECREF()
 * for "borrowed references" (with objects from certain python calls).
 *
 * NOTE2: You should probably call PyEval_SaveThread() 
 * before GIL lock and also after GIL release PyEval_RestoreThread() to restore thread state
 * in EVERY function that makes python calls. and not just in constructor and destructor.
 */

#include "MinihackRIFL.h"


namespace whiteice
{
  
  // observation space size is 51 (5x5 char environment + player stats) [and action is one-hot-encoded value: NOT]
  template <typename T>
  MinihackRIFL<T>::MinihackRIFL(const std::string& pythonScript) 
    : RIFL_abstract3<T>(8, 51, {100,100,100,100,100})
  // : RIFL_abstract3<T>(8, 51, {50,50,50})
    // : RIFL_abstract3<T>(8, 51, {50,50,50,50,50})
  // : RIFL_abstract<T>(8, 51, {200,200,200,200})
  {
    // we inteprete action values as one hot encoded probabilistic values from which one-hot-encoded
    // vector is chosen: [0 0 1 0] means 3rd action is chosen.
    //this->setOneHotAction(true);
    //this->setSmartEpisodes(true); // gives more weight to reinforcement values when calculating Q
    
    
    if(!Py_IsInitialized()){
      Py_Initialize();
    }

    // PyEval_InitThreads();
    auto pystate = PyEval_SaveThread();

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    errors = 0;

    //PyThreadState* prestate = PyThreadState_Get();
    //pystate = PyThreadState_New(prestate->interp);
    //
    //if(pystate == NULL){
    //  errors++;
    //  assert(0); // FIXME do proper error handling
    //}    

    // PyEval_RestoreThread(pystate);

    {
      int argc = 1;
      wchar_t* argv[1];
      const char* program = "minihack";
      const unsigned int cSize = strlen(program)+1;
      wchar_t* ptr = (wchar_t*)malloc(sizeof(wchar_t)*cSize);

      mbstowcs (ptr, program, cSize);
      argv[0] = ptr;
      
      PySys_SetArgv(argc, argv); // fix??

      free(ptr);
    }

    // PyRun_SimpleString("x = 0"); // dummy (needed?)

    pythonFile = fopen(pythonScript.c_str(), "r");

    if(pythonFile == NULL){
      errors++;
      assert(0); // FIXME proper error handling
    }

    filename = pythonScript;

    PyRun_SimpleFile(pythonFile, filename.c_str());

    main_module = PyImport_AddModule("__main__");
    global_dict = PyModule_GetDict(main_module);

    if(main_module == NULL || global_dict == NULL){
      errors++;
      assert(0); // FIXME proper error handling
    }

    getStateFunc = PyDict_GetItemString(global_dict, (const char*)"minihack_getState");
    performActionFunc = PyDict_GetItemString(global_dict, (const char*)"minihack_performAction");

    if(getStateFunc == NULL || performActionFunc == NULL){
      errors++;
      assert(0); // FIXME proper error handling
    }

    //pystate = PyEval_SaveThread();
    //PyEval_RestoreThread(prestate);

    PyGILState_Release(gstate);
    PyEval_RestoreThread(pystate);
  }
  

  template <typename T>
  MinihackRIFL<T>::~MinihackRIFL()
  {
    //PyThreadState* prestate = PyThreadState_Get();
    //PyEval_RestoreThread(pystate);

    auto pystate = PyEval_SaveThread();

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    
    Py_DECREF(getStateFunc);
    Py_DECREF(performActionFunc);
    //Py_DECREF(global_dict);
    //Py_DECREF(main_module);

    //PyEval_RestoreThread(prestate);
    //
    //if(pystate){
    //  PyThreadState_Clear(pystate);
    //  PyThreadState_Delete(pystate);
    //}
    //
    //pystate = NULL;

    PyGILState_Release(gstate);
    PyEval_RestoreThread(pystate);

#if 0
    if(pystate){
      PyEval_RestoreThread(pystate);
      pystate = NULL;
    }
#endif

    // don't finalize python state
    Py_Finalize();
    
    if(pythonFile) fclose(pythonFile);
  }

  
  template <typename T>
  bool MinihackRIFL<T>::isRunning() const
  {
    return (errors == 0);
  }
  
  
  template <typename T>
  bool MinihackRIFL<T>::getState(whiteice::math::vertex<T>& state)
  {
    if(errors > 0) return false;

    auto pystate = PyEval_SaveThread();

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    //PyThreadState* prestate = PyThreadState_Get();
    //PyEval_RestoreThread(pystate);
    
    PyObject *result = NULL;
    
    result = PyObject_CallFunction(getStateFunc, NULL);

    if(result == NULL){
      printf("ERROR: getState(): PyObject_CallFunction() returned NULL.\n");
      errors++;
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      return false;
    }

    if(PyList_CheckExact(result) == 0){
      printf("ERROR: getState(): PyObject_CallFunction() returned non-list.\n");
      errors++;
      Py_DECREF(result);
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      return false;
    }

    const unsigned long SIZE = (unsigned long)PyList_Size(result);

    if(SIZE != this->getNumStates()){
      printf("ERROR: getState(): returned state has wrong length.\n");
      errors++;
      Py_DECREF(result);
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      return false;
    }

    if(SIZE > 0){

      if(state.resize(SIZE) == false){
	printf("ERROR: getState(): state.resize() FAILED.\n");
	errors++;
	Py_DECREF(result);
	//pystate = PyEval_SaveThread();
	//PyEval_RestoreThread(prestate);
	PyGILState_Release(gstate);
	PyEval_RestoreThread(pystate);
	return false;
      }

      for(unsigned long index=0;index<SIZE;index++){
	PyObject* item = PyList_GetItem(result, (Py_ssize_t)index);

	if(item == NULL){
	  printf("ERROR: getState(): returned list contains NULL.\n");
	  errors++;
	  Py_DECREF(result);
	  
	  //pystate = PyEval_SaveThread();
	  //PyEval_RestoreThread(prestate);
	  PyGILState_Release(gstate);
	  PyEval_RestoreThread(pystate);
	  
	  return false;	  
	}

	if(PyLong_CheckExact(item) == 0){
	  printf("ERROR: getState(): list item is not long.\n");
	  errors++;
	  //Py_DECREF(item);
	  Py_DECREF(result);
	  
	  //pystate = PyEval_SaveThread();
	  //PyEval_RestoreThread(prestate);
	  PyGILState_Release(gstate);
	  PyEval_RestoreThread(pystate);
	  
	  return false;
	}

	state[index] = T(PyLong_AsDouble(item));		

	//Py_DECREF(item);
      }
    }

    Py_DECREF(result);

    //pystate = PyEval_SaveThread();
    //PyEval_RestoreThread(prestate);
    PyGILState_Release(gstate);
    PyEval_RestoreThread(pystate);

    return true;
  }

  
  template <typename T>
  bool MinihackRIFL<T>::performAction(const unsigned int action,
				       whiteice::math::vertex<T>& newstate,
				       T& reinforcement, bool& endFlag)
  {
    if(errors > 0){
      printf("ERROR: performAction(), errors>0\n");
      return false;
    }
    
    // [state, reward, done] = minihack_performAction(action) (action is integer 0..7)

    if(action >= this->getNumActions()){
      printf("ERROR: performAction(): unknown/bad action value.\n");
      errors++;
      return false;
    }

    const unsigned long ACTION = (unsigned long)action;

    //printf("ACTION %d selected.\n", (int)ACTION); fflush(stdout);

    //PyThreadState* prestate = PyThreadState_Get();
    //PyEval_RestoreThread(pystate);

    auto pystate = PyEval_SaveThread();
    
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    
    PyObject *result = NULL;

    //printf("PyObject_CallFunction() = %p.\n", (void*)performActionFunc); fflush(stdout);
    
    result = PyObject_CallFunction(performActionFunc, "k", (unsigned long)ACTION);

    //printf("PyObject_CallFunction() called.\n"); fflush(stdout);

    if(result == NULL){
      printf("ERROR: performAction(): PyObject_CallFunction() returned NULL.\n");
      errors++;
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    //printf("PyObject_CallFunction() returned: %p.\n", (void*)result); fflush(stdout);
    
    // [state, reward, done] = minihack_performAction(action)
    // there are now multiple return values state (int list to double list), reward is float, done is boolean flag

    
    // check return value is a list with 3 elements

    if(PyList_CheckExact(result) == 0){
      printf("ERROR: performAction(): PyObject_CallFunction() don't return list.\n");
      errors++;
      Py_DECREF(result);
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    if(PyList_Size(result) != 3){
      printf("ERROR: performAction(): PyObject_CallFunction() return value length is not 3.\n");
      errors++;
      Py_DECREF(result);
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    // extract return values from the list

    PyObject* stateObj = PyList_GetItem(result, 0);
    PyObject* rewardObj = PyList_GetItem(result, 1);
    PyObject* doneObj = PyList_GetItem(result, 2);

    if(stateObj == NULL || rewardObj == NULL || doneObj == NULL){
      printf("ERROR: performAction(): PyObject_CallFunction() return objects are NULL.\n");
      errors++;

      //if(stateObj) Py_DECREF(stateObj);
      //if(rewardObj) Py_DECREF(rewardObj);
      //if(doneObj) Py_DECREF(doneObj);
      
      Py_DECREF(result);
      
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    //printf("Return objects loaded (%p, %p, %p).\n", (void*)stateObj, (void*)rewardObj, (void*)doneObj); fflush(stdout);


    if(PyList_CheckExact(stateObj) == 0){
      printf("ERROR: performAction(): state object is not list.\n");
      errors++;
      
      //Py_DECREF(stateObj);
      //Py_DECREF(rewardObj);
      //Py_DECREF(doneObj);
      
      Py_DECREF(result);

      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }    

    const unsigned long SIZE = (unsigned long)PyList_Size(stateObj);

    if(SIZE != this->getNumStates()){
      printf("ERROR: performAction(): state object doesn't have proper length.\n");
      errors++;
      
      Py_DECREF(result);
      
      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    //printf("List object seen (SIZE: %d).\n", (int)SIZE); fflush(stdout);

    if(SIZE > 0){

      if(newstate.resize(SIZE) == false){
	printf("ERROR: performAction(): newstate.resize() FAILED.\n");
	errors++;
	
	//Py_DECREF(stateObj);
	//Py_DECREF(rewardObj);
	//Py_DECREF(doneObj);
	
	Py_DECREF(result);

	//pystate = PyEval_SaveThread();
	//PyEval_RestoreThread(prestate);
	PyGILState_Release(gstate);
	PyEval_RestoreThread(pystate);
	
	return false;
      }

      for(unsigned long index=0;index<SIZE;index++){
	PyObject* item = PyList_GetItem(stateObj, (Py_ssize_t)index);

	if(item == NULL){
	  printf("ERROR: performAction(): state list item is NULL.\n");
	  errors++;
	  
	  //Py_DECREF(stateObj);
	  //Py_DECREF(rewardObj);
	  //Py_DECREF(doneObj);
	  
	  Py_DECREF(result);

	  //pystate = PyEval_SaveThread();
	  //PyEval_RestoreThread(prestate);
	  PyGILState_Release(gstate);
	  PyEval_RestoreThread(pystate);

	  return false;
	}

	if(PyLong_CheckExact(item) == 0){
	  printf("ERROR: performAction(): state list item not LONG.\n");
	  errors++;
	  
	  //Py_DECREF(item);

	  //Py_DECREF(stateObj);
	  //Py_DECREF(rewardObj);
	  //Py_DECREF(doneObj);
	  
	  Py_DECREF(result);

	  //pystate = PyEval_SaveThread();
	  //PyEval_RestoreThread(prestate);
	  PyGILState_Release(gstate);
	  PyEval_RestoreThread(pystate);

	  return false;
	}
	
	newstate[index] = T(PyLong_AsDouble(item));
	
	//Py_DECREF(item);
      }
    }

    //printf("New state object loaded.\n"); fflush(stdout);

    //printf("1. rewardObj = %p.\n", (void*)rewardObj); fflush(stdout);

    if(PyFloat_Check(rewardObj) == 0){
      printf("ERROR: performAction(): reward object is not float.\n");
      errors++;

      printf("rewardObj check FAILED.\n"); fflush(stdout);
      
      //Py_DECREF(stateObj);
      //Py_DECREF(rewardObj);
      //Py_DECREF(doneObj);
    
      Py_DECREF(result);

      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    //printf("2. rewardObj OK = %p.\n", (void*)rewardObj); fflush(stdout);

    reinforcement = T(PyFloat_AsDouble(rewardObj));

    //printf("Reinforcment value loaded: %f.\n", reinforcement.c[0]); fflush(stdout);
    
    if(PyBool_Check(doneObj) == 0){
      printf("ERROR: performAction(): done object is not bool.\n");
      errors++;
      
      //Py_DECREF(stateObj);
      //Py_DECREF(rewardObj);
      //Py_DECREF(doneObj);
    
      Py_DECREF(result);

      //pystate = PyEval_SaveThread();
      //PyEval_RestoreThread(prestate);
      PyGILState_Release(gstate);
      PyEval_RestoreThread(pystate);
      
      return false;
    }

    if(doneObj == Py_False)
      endFlag = false;
    else
      endFlag = true;

    //printf("Done flag loaded: %d.\n", endFlag); fflush(stdout);
    
    //Py_DECREF(stateObj);
    //Py_DECREF(rewardObj);
    //Py_DECREF(doneObj);
    
    Py_DECREF(result);

    //pystate = PyEval_SaveThread();
    //PyEval_RestoreThread(prestate);
    PyGILState_Release(gstate);
    PyEval_RestoreThread(pystate);

    return true;
  }
  
  

  template class MinihackRIFL< math::blas_real<float> >;
  template class MinihackRIFL< math::blas_real<double> >;  
  
};

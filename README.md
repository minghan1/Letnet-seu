Organization of the Repo:
（1）create lenet modle: create_model.py and create_seumodel.py. create_seumodel.py is intended for testing the radiation-hardened capability of a modified training method or altered network structure.However, it is currently not yet operational.
  (2)  fault injection : fault_injection.py seu.py
  (3)  Top-level-testing : seutest.py seutest_noseu.py seutest2.py seutest3.py. seutest_noseu also cannot be used, for same reason as create_seumodel.
  (4) Data processing : dealdata.py  extem_p.py get_model_extrem_num.py weights_distribution.py.  In fact, much of our data processing is done directly in the top-level testing. Here is some code for obtaining the probabilities mentioned in the paper.
  (5)  Results and others : The remaining
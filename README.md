# RACoon
RACoon : Relational DNN Verification With Cross Executional Bound Refinement

The relational verifier runs through the unit test framework. A new unit test can be added to run the verifier
with a specific configuration. 
Current unit tests are located in `src/tests/` folder. 

## Reproducing Experiments

Move to the src directory

```
cd src
```

Then run the following any of the following commands 

### Untargeted UAP Verification

Run any experiment by replacing ``test_file_name``, ``test_class_name`` and ``test_name`` from the folder ``src/tests/``

```
python3 -m unittest -v src.tests.test_file_name.test_class_name.test_name
```

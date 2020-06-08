TEXT_DATA="test"
TEST_DATA_PATH="./data/test.pkl"
BEST_MODEL_PATH="./crossModal_model/best_model/model.pt"
ARGS_PATH="./crossModal_model/best_model/args.pt"

python test.py --text_data=${TEXT_DATA} \
    --test_data_path=${TEST_DATA_PATH} \
    --best_model_path=${BEST_MODEL_PATH} \
    --args_path=${ARGS_PATH}
    


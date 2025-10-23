import sys
sys.path.insert(0, 'src')
from bedrock_benchmark.storage import StorageManager
from bedrock_benchmark.models import BenchmarkItem, BedrockResponse, RunConfig
from datetime import datetime
import tempfile
import os

# Test the DataFrame export functionality
with tempfile.TemporaryDirectory() as temp_dir:
    # Create storage manager
    storage = StorageManager(temp_dir)
    
    # Create experiment and run
    exp_id = storage.create_experiment('Test Experiment', 'Testing DataFrame export')
    config = RunConfig(model_id='anthropic.claude-3-sonnet-20240229-v1:0', dataset_path='test.jsonl')
    run_id = storage.create_run(exp_id, config)
    
    # Create test data
    item = BenchmarkItem(id='item1', prompt='Test prompt', expected_response='Expected response')
    response = BedrockResponse(
        item_id='item1',
        response_text='Actual response',
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        timestamp=datetime.now(),
        latency_ms=500,
        input_tokens=10,
        output_tokens=15,
        finish_reason='stop'
    )
    
    # Save response
    storage.save_response(run_id, response)
    
    # Test single run export
    df = storage.export_run_to_dataframe(run_id, [item])
    print('Single run DataFrame shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Sample data:')
    print(df[['run_id', 'prompt', 'expected_response', 'actual_response', 'model_id']].head())
    
    # Test multiple runs export
    df_multi = storage.export_multiple_runs_to_dataframe([run_id], [item])
    print('\nMultiple runs DataFrame shape:', df_multi.shape)
    
    # Test run summary
    summary = storage.get_run_summary(run_id)
    print('\nRun summary:', summary)
    
    print('\nDataFrame export functionality test completed successfully!')
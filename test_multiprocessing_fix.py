#!/usr/bin/env python3
"""
Test script to verify that the multiprocessing pickle error is fixed
"""

import os
import sys
from multiprocessing import Pool
from functools import partial

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.datasets.open_world_running_dataset import OpenWorldRunningDataset
from data.datasets.cache import Cache
from s3.s3_utils import S3Manager

def test_pickle_fix():
    """Test that we can now pickle the function with parameters"""
    
    # Mock S3Manager parameters (these should be picklable)
    s3_manager_params = {
        'bucket_name': 'test-bucket',
        'region_name': 'us-east-1',
        'aws_access_key_id': 'test-key',
        'aws_secret_access_key': 'test-secret',
    }
    
    # Mock Cache parameters (these should be picklable)
    local_cache_params = {
        'max_size': 1000,
        'cache_dir': '/tmp/test-cache',
    }
    
    # Test data
    chunks = []  # Empty chunks for testing
    
    # Create the partial function that was causing the pickle error
    partial_fn = partial(
        OpenWorldRunningDataset._get_frame_sequences,
        limit=10,
        max_frame_sequence_length=100,
        num_frames_in_video=30,
        s3_manager_params=s3_manager_params,
        local_cache_params=local_cache_params,
    )
    
    # Try to pickle the function (this is what multiprocessing does internally)
    import pickle
    try:
        pickled_fn = pickle.dumps(partial_fn)
        print("✅ SUCCESS: Function can be pickled!")
        
        # Try to unpickle it
        unpickled_fn = pickle.loads(pickled_fn)
        print("✅ SUCCESS: Function can be unpickled!")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Cannot pickle function: {e}")
        return False

if __name__ == "__main__":
    print("Testing multiprocessing pickle fix...")
    success = test_pickle_fix()
    
    if success:
        print("\n🎉 All tests passed! The multiprocessing pickle error should be fixed.")
    else:
        print("\n💥 Tests failed. The pickle error may still exist.")
    
    sys.exit(0 if success else 1)


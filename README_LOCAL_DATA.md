# Local .npy Data Loading for H100 Training

I've updated the training pipeline to use your local pre-tokenized .npy files instead of downloading OpenWebText. This provides several key advantages:

## ✅ **Benefits of Local Data Loading**

1. **No Download Time**: Skip the multi-hour OpenWebText download
2. **Faster I/O**: Memory-mapped .npy files are extremely efficient
3. **Consistent Data**: Same tokenization across runs
4. **No Network Dependencies**: Fully offline training
5. **Better Memory Management**: Efficient streaming from disk

## 📁 **Data Files Used**

Your `training_data/` folder contains:
- `owt_train_tokens.npy` (5.1GB) - Training data
- `owt_valid_tokens.npy` (127MB) - Validation data

These files contain pre-tokenized OpenWebText using GPT-2 tokenizer.

## 🔧 **Key Changes Made**

### 1. **New Dataset Classes**

**`LocalTokenizedDataset` (IterableDataset)**:
- Memory-mapped access to .npy files
- Efficient multi-worker data loading
- Proper shuffling and worker distribution
- Streaming interface for large datasets

**`LocalTokenizedDatasetFixed` (Dataset)**:
- Fixed-size dataset with indexing
- Even more efficient for smaller datasets
- Direct random access to sequences

### 2. **Updated Data Module**

**`LocalDataModule`**:
- Handles both train and validation .npy files
- Configurable paths for different data sources
- Optimized for H100 throughput

### 3. **Enhanced create_dataloaders() Function**

```python
train_loader, val_loader = create_dataloaders(
    train_data_path="training_data/owt_train_tokens.npy",
    val_data_path="training_data/owt_valid_tokens.npy",
    batch_size=32,
    max_length=1024,
    num_workers=8,
    prefetch_factor=4,
    seed=42,
)
```

## 🚀 **Usage**

### **Test Data Loading**
```bash
python test_local_data.py
```

### **Run Optimized Training**
```bash
python train_h100_optimized.py
```

### **Custom Data Paths**
```bash
python train_h100_optimized.py \
    --train_data_path /path/to/your/train.npy \
    --val_data_path /path/to/your/val.npy \
    --batch_size 64
```

## ⚡ **Performance Benefits**

| Aspect | Before (OpenWebText download) | After (Local .npy) |
|--------|-------------------------------|---------------------|
| **Setup Time** | 2-4 hours download | Instant |
| **I/O Speed** | Network + HuggingFace parsing | Memory-mapped .npy |
| **Memory Usage** | High (tokenization overhead) | Low (pre-tokenized) |
| **Consistency** | Varies by download | Always same data |
| **Reliability** | Network dependent | 100% offline |

## 🔍 **Data Loading Process**

1. **Memory Mapping**: Files are memory-mapped for efficient access
2. **Worker Distribution**: Each DataLoader worker gets a portion of data
3. **Shuffling**: Proper shuffling while maintaining efficiency
4. **Batching**: Optimized collation for maximum throughput

## 📊 **Expected Data Statistics**

With your files:
- **Training tokens**: ~1.3B tokens (5.1GB ÷ 4 bytes)
- **Validation tokens**: ~32M tokens (127MB ÷ 4 bytes)
- **Training sequences**: ~1.2M sequences (1.3B ÷ 1024)
- **Validation sequences**: ~31K sequences (32M ÷ 1024)

## 🛠️ **Technical Implementation**

### **Memory-Mapped Access**
```python
self.tokens = np.memmap(data_path, dtype=np.int32, mode='r')
```

### **Efficient Sequence Extraction**
```python
start_idx = seq_idx * self.max_length
end_idx = start_idx + self.max_length
sequence = torch.from_numpy(self.tokens[start_idx:end_idx].astype(np.int64))
```

### **Multi-Worker Support**
- Each worker gets a unique portion of sequences
- Proper shuffling within worker ranges
- No data overlap between workers

## 🔥 **Expected Performance Improvement**

With local .npy files, you should see:
- **Faster Training Start**: No download wait time
- **Better I/O Performance**: Memory-mapped files are extremely fast
- **More Consistent Performance**: No network variability
- **Higher GPU Utilization**: Less time waiting for data

This should help you achieve the target validation loss of **2.8-2.9** (beating 3.0781) more reliably and faster on your H100!

## 🐛 **Troubleshooting**

### **File Not Found**
```bash
❌ Training data file not found: training_data/owt_train_tokens.npy
```
- Ensure files are in the correct location
- Check file permissions

### **Memory Issues**
If you encounter memory issues:
- Reduce `num_workers`
- Reduce `prefetch_factor`
- Reduce `batch_size`

### **Performance Issues**
For optimal performance:
- Use SSD storage for .npy files
- Increase `num_workers` (up to 8-16)
- Increase `prefetch_factor` (up to 4-8)
- Use larger batch sizes on H100

## 📝 **Files Modified**

1. `cs336_basics/data/dataloader.py` - New local data classes
2. `cs336_basics/data/__init__.py` - Updated exports
3. `train_h100_optimized.py` - Added data path arguments
4. `test_local_data.py` - New test script

Your training should now be much faster and more reliable! 🚀
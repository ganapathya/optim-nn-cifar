# 🚀 Huggingface Spaces Deployment Guide

Complete guide to deploy your CIFAR-100 ResNet-18 classifier to Huggingface Spaces.

## 📋 Prerequisites

1. **Huggingface Account** - Create one at [huggingface.co](https://huggingface.co/join)
2. **Git** - Installed on your system
3. **Trained Model** - ✅ Already have `best_model.pth` (78.33% accuracy!)

## 🎯 Deployment Methods

### Method 1: Web UI Upload (Easiest) ⭐

#### Step 1: Create New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in details:

   - **Space name:** `cifar100-resnet18-classifier` (or your choice)
   - **License:** MIT
   - **Select SDK:** Gradio
   - **SDK version:** 4.0.0 (or latest)
   - **Space hardware:** CPU Basic (free) - sufficient for inference
   - **Visibility:** Public

3. Click **Create Space**

#### Step 2: Upload Files

In your new Space, click **Files** → **Add file** → **Upload files**

Upload these files from `gradio_app/` folder:

**Required Files:**

```
✅ app.py                    # Main application
✅ requirements.txt          # Dependencies
✅ best_model.pth           # Trained model (86MB)
✅ cifar100/                # Module folder (upload entire folder)
   ├── __init__.py
   ├── model.py
   ├── utils.py
   ├── inference.py
   └── config.py
```

**Optional but Recommended:**

```
✅ README_HF.md             # Rename to README.md when uploading
```

#### Step 3: Configure README

1. Rename `README_HF.md` to `README.md` before uploading
2. This will display nice documentation on your Space

#### Step 4: Wait for Build

- Huggingface will automatically:
  - Install dependencies from `requirements.txt`
  - Build the app
  - Start the Gradio interface
- Build time: ~2-3 minutes
- Status visible at top of page

#### Step 5: Access Your App! 🎉

Your app will be live at:

```
https://huggingface.co/spaces/YOUR_USERNAME/cifar100-resnet18-classifier
```

---

### Method 2: Git Push (Advanced)

#### Step 1: Create Space (same as Method 1)

Create your Space on Huggingface website first.

#### Step 2: Clone Your Space

```bash
# Install Git LFS (for large files)
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME
```

#### Step 3: Copy Files

From your project directory:

```bash
# Copy all necessary files to Space directory
cp /Users/ganapathysubramani/workspace/optim-nn-sifar/gradio_app/app.py .
cp /Users/ganapathysubramani/workspace/optim-nn-sifar/gradio_app/requirements.txt .
cp /Users/ganapathysubramani/workspace/optim-nn-sifar/gradio_app/best_model.pth .
cp /Users/ganapathysubramani/workspace/optim-nn-sifar/gradio_app/README_HF.md ./README.md
cp -r /Users/ganapathysubramani/workspace/optim-nn-sifar/gradio_app/cifar100 .
```

#### Step 4: Track Large Files with Git LFS

```bash
# Track the model file with LFS
git lfs track "*.pth"
git add .gitattributes
```

#### Step 5: Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment: ResNet-18 CIFAR-100 classifier (78.33% accuracy)"

# Push to Huggingface
git push
```

#### Step 6: Monitor Deployment

Watch the build logs on your Space page. App will be live in ~2-3 minutes.

---

## 📦 Files Checklist

Before deploying, verify you have:

```
gradio_app/
├── app.py                  ✅ Updated for HF Spaces
├── requirements.txt        ✅ All dependencies listed
├── best_model.pth         ✅ 86MB, 78.33% accuracy
├── README_HF.md           ✅ Documentation (rename to README.md)
└── cifar100/              ✅ Complete module
    ├── __init__.py
    ├── model.py
    ├── utils.py
    ├── inference.py
    └── config.py
```

## 🔧 Configuration Options

### Change Space Visibility

In Space settings:

- **Public** - Anyone can access
- **Private** - Only you can access

### Upgrade Hardware (Optional)

Free CPU Basic tier is sufficient, but for faster inference:

- **CPU Upgrade** - $0.03/hour (2x faster)
- **T4 GPU** - $0.60/hour (10x faster)

### Enable Gradio Queue

For multiple users, add to `app.py`:

```python
interface.queue().launch()
```

## 🧪 Testing Your Deployed App

### 1. Open Your Space URL

```
https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
```

### 2. Test Upload

- Drag and drop an image
- Or click to browse files
- Should see predictions instantly

### 3. Verify Results

Expected behavior:

- Upload shows preview
- Model processes (~1 second)
- Top 5 predictions display with confidence scores
- Classes formatted nicely (e.g., "Sweet Pepper" not "sweet_pepper")

### 4. Check Examples

Try these image types:

- **Animal photo** → Should predict animal class
- **Car/vehicle** → Should predict vehicle type
- **Nature scene** → Should predict landscape class
- **Object photo** → Should predict object type

## 📊 Expected Performance

### Model Stats

- **Accuracy:** 78.33% on CIFAR-100 test set
- **Parameters:** 11,220,132
- **Model Size:** 86MB
- **Inference Time:** ~10-20ms per image on CPU

### Space Stats

- **Build Time:** 2-3 minutes
- **Cold Start:** ~3-5 seconds (first request)
- **Warm Requests:** <100ms
- **Concurrent Users:** 1 (free tier), more with queue

## 🐛 Troubleshooting

### "Application failed to start"

**Check:**

1. All files uploaded correctly
2. `requirements.txt` has correct versions
3. `best_model.pth` uploaded completely (86MB)
4. `cifar100/` folder structure intact

**Fix:**

- Re-upload missing files
- Check Space build logs
- Verify no file corruption

### "Model not loaded" Error

**Cause:** Model path incorrect or file missing

**Fix:**

```python
# In app.py, verify path:
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
```

### Slow Inference

**Normal:** First request takes 3-5 seconds (cold start)

**If always slow:**

- Upgrade to faster hardware
- Enable Gradio queue
- Check Space resources in settings

### Import Errors

**Cause:** Missing dependencies

**Fix:** Update `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
numpy>=1.24.0
albumentations>=1.3.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
```

## 🎨 Customization Ideas

### Add Example Images

1. Create `examples/` folder in Space
2. Add sample images
3. Update `app.py`:

```python
examples=[
    ["examples/cat.jpg"],
    ["examples/car.jpg"],
    ["examples/tree.jpg"]
]
```

### Change Theme

In `app.py`:

```python
theme=gr.themes.Glass()  # or Base(), Monochrome(), etc.
```

### Add Analytics

Track usage with Huggingface Analytics (automatic in Spaces)

### Custom Domain

In Space settings, you can add a custom domain

## 📈 Post-Deployment

### Share Your Space

- Tweet the link
- Add to your portfolio
- Share in ML communities
- Link from GitHub README

### Monitor Usage

Check Space metrics:

- Number of visitors
- API calls
- Resource usage

### Iterate

Based on feedback:

- Add more example images
- Improve descriptions
- Update model (retrain and replace)
- Add new features

## 🎉 Success Criteria

Your deployment is successful when:

✅ Space builds without errors  
✅ App loads in browser  
✅ Image upload works  
✅ Predictions display correctly  
✅ Accuracy matches training (78.33%)  
✅ Documentation displays nicely  
✅ No console errors

## 📞 Getting Help

**Resources:**

- [Huggingface Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs)
- [Community Forum](https://discuss.huggingface.co)

**Common Issues:**

- Check Space build logs
- Verify file structure
- Test locally first: `python app.py`

---

## 🚀 Quick Start Commands

**Test Locally First:**

```bash
cd gradio_app
python app.py
# Open http://localhost:7860
```

**Everything Works?** → Upload to Huggingface Spaces!

**Need Help?** Check the troubleshooting section above.

---

**Ready to deploy? Follow Method 1 above for the easiest approach!** 🎉

Good luck with your deployment!

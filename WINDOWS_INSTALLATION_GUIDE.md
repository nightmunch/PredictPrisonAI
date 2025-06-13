# Windows Installation Guide
## Malaysia Prison Predictive Planning System

### Quick Start (3 Steps)

1. **Download Python**
   - Go to https://python.org/downloads/
   - Download Python 3.11 (latest version)
   - **IMPORTANT**: Check "Add Python to PATH" during installation

2. **Download Application Files**
   - Download all application files to a folder (e.g., `C:\PrisonForecasting`)
   - Right-click `setup_windows.bat` → "Run as administrator"

3. **Launch Application**
   - Double-click `run_app.bat`
   - Application opens in your web browser at http://localhost:8501

### Manual Installation Steps

If the batch files don't work, follow these manual steps:

#### Step 1: Install Python
1. Download Python 3.11 from python.org
2. Run installer with these settings:
   - ✅ Add Python to PATH
   - ✅ Install pip
   - Choose "Customize installation"
   - ✅ Install for all users

#### Step 2: Install Dependencies
Open Command Prompt as Administrator and run:
```
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
```

#### Step 3: Run Application
Navigate to your application folder and run:
```
streamlit run app.py
```

### Folder Structure for Windows

Create this folder structure on your Windows system:
```
C:\PrisonForecasting\
├── app.py
├── modules\
├── utils\
├── models\
├── .streamlit\
├── setup_windows.bat
├── run_app.bat
└── README.md
```

### Troubleshooting Windows Issues

**Python not found:**
- Uninstall and reinstall Python with "Add to PATH" checked
- Restart Command Prompt

**Permission errors:**
- Run Command Prompt as Administrator
- Right-click batch files → "Run as administrator"

**Port conflicts:**
- Change port: `streamlit run app.py --server.port 8502`

**Antivirus blocking:**
- Add Python and application folder to antivirus exceptions

### Windows-Specific Features

- **Desktop Shortcut**: Create shortcut to `run_app.bat` on desktop
- **Startup**: Add to Windows startup folder for auto-start
- **Windows Service**: Can be configured to run as Windows service
- **Task Scheduler**: Schedule automatic model retraining

### Performance on Windows

- **Recommended**: 8GB RAM, Intel i5 or equivalent
- **Minimum**: 4GB RAM, dual-core processor
- **Storage**: 2GB free space for data and models
- **Browser**: Chrome, Firefox, or Edge (latest versions)

### Security Considerations

- Application runs locally only
- No internet connection required after installation
- All data stays on your machine
- Suitable for sensitive government data
- Windows Defender compatible

### Backup Strategy

Important folders to backup regularly:
- `data\` - Historical datasets
- `models\` - Trained AI models
- `.streamlit\config.toml` - Application settings

### Updates and Maintenance

Monthly maintenance:
```
pip install --upgrade streamlit pandas numpy scikit-learn plotly
```

Quarterly model retraining:
- Use "Retrain Models" button in Model Performance section
- Or delete `models\` folder and restart application

---

**Support**: For Windows-specific issues, ensure you have administrator privileges and the latest Python version installed.
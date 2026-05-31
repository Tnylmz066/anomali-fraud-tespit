# Anomali ve Fraud Tespit Sistemi - otomatik kurulum
# Python (3.9-3.11) bulur ya da kurar, izole .venv olusturur,
# paketleri yukler ve masaustu kisayolunu olusturur.
param(
    [string]$ShortcutDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $ShortcutDir) { $ShortcutDir = [Environment]::GetFolderPath("Desktop") }

Write-Host ""
Write-Host "======================================================"
Write-Host "  ANOMALI VE FRAUD TESPIT SISTEMI - KURULUM"
Write-Host "======================================================"
Write-Host ""

function Find-Python {
    # TensorFlow 2.15 icin 3.9 - 3.11 gerekli (3.12+ desteklenmez)
    foreach ($v in @("3.11", "3.10", "3.9")) {
        try {
            $exe = & py -$v -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($ver -match '^3\.(9|10|11)$') {
                return (& python -c "import sys; print(sys.executable)").Trim()
            }
        } catch {}
    }
    return $null
}

# 1) Python bul
Write-Host "[1/5] Uygun Python (3.9-3.11) araniyor..."
$py = Find-Python

if (-not $py) {
    Write-Host "      Uygun Python bulunamadi. winget ile Python 3.11 kurulmaya calisiliyor..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        try {
            & winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        } catch {
            Write-Host "      winget kurulumu sirasinda uyari olustu, devam ediliyor..."
        }
        Start-Sleep -Seconds 3
        $py = Find-Python
    } else {
        Write-Host "      winget bulunamadi."
    }
}

if (-not $py) {
    Write-Host ""
    Write-Host "HATA: Uygun bir Python kurulamadi."
    Write-Host "Lutfen https://www.python.org/downloads/release/python-3119/ adresinden"
    Write-Host "Python 3.11 kurun (kurulumda 'Add to PATH' isaretli olsun) ve KUR.bat'i tekrar calistirin."
    Write-Host "winget yeni kurduysa: bu pencereyi kapatip KUR.bat'i bir kez daha calistirin."
    exit 1
}
Write-Host "      Python bulundu: $py"

# 2) venv olustur
Write-Host "[2/5] Izole ortam (.venv) hazirlaniyor..."
$venv = Join-Path $root ".venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $vpy)) {
        Write-Host "HATA: Sanal ortam olusturulamadi."
        exit 1
    }
} else {
    Write-Host "      Mevcut .venv kullanilacak."
}

# 3) pip ayarlari (bazi aglarda SSL sorununu asmak icin)
$pipIni = Join-Path $venv "pip.ini"
@"
[global]
trusted-host = pypi.org
    files.pythonhosted.org
    pypi.python.org
"@ | Set-Content -Path $pipIni -Encoding ASCII

# 4) Paketleri yukle
Write-Host "[3/5] pip guncelleniyor..."
& $vpy -m pip install --upgrade pip
Write-Host "[4/5] Paketler yukleniyor (TensorFlow dahil - ilk seferde birkac dakika surebilir)..."
& $vpy -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "HATA: Paketler yuklenemedi. Internet baglantinizi kontrol edip tekrar deneyin."
    exit 1
}

# 5) Masaustu kisayolu
Write-Host "[5/5] Masaustu kisayolu olusturuluyor..."
$pyw = Join-Path $venv "Scripts\pythonw.exe"
$mainPy = Join-Path $root "app\main.py"
$appDir = Join-Path $root "app"
$icon = Join-Path $root "app\app_icon.ico"

if (-not (Test-Path $ShortcutDir)) { New-Item -ItemType Directory -Path $ShortcutDir -Force | Out-Null }
$lnkPath = Join-Path $ShortcutDir "Anomali ve Fraud Tespit.lnk"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath = $pyw
$lnk.Arguments = '"' + $mainPy + '"'
$lnk.WorkingDirectory = $appDir
if (Test-Path $icon) { $lnk.IconLocation = $icon }
$lnk.Description = "Anomali ve Fraud Tespit Sistemi"
$lnk.WindowStyle = 1
$lnk.Save()

Write-Host ""
Write-Host "======================================================"
Write-Host "  KURULUM TAMAMLANDI"
Write-Host "======================================================"
Write-Host "  Masaustundeki 'Anomali ve Fraud Tespit' kisayolundan"
Write-Host "  uygulamayi baslatabilirsiniz."
Write-Host "======================================================"
Write-Host ""

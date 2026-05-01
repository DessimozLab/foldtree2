@echo off

:: Install the package using pip
%PYTHON% -m pip install . -vv
if errorlevel 1 exit 1

:: Copy external tools
mkdir %PREFIX%\share\foldtree2 2>nul

:: Copy bundled tools from package source tree
if exist foldtree2\raxml-ng (
    xcopy /E /I foldtree2\raxml-ng %PREFIX%\share\foldtree2\raxml-ng
    if exist %PREFIX%\share\foldtree2\raxml-ng\raxml-ng.exe (
        copy %PREFIX%\share\foldtree2\raxml-ng\raxml-ng.exe %PREFIX%\Scripts\raxml-ng.exe
    )
)

if exist foldtree2\madroot (
    xcopy /E /I foldtree2\madroot %PREFIX%\share\foldtree2\madroot
    if exist %PREFIX%\share\foldtree2\madroot\mad.exe (
        copy %PREFIX%\share\foldtree2\madroot\mad.exe %PREFIX%\Scripts\mad.exe
    )
)

if exist foldtree2\mafft_tools (
    xcopy /E /I foldtree2\mafft_tools %PREFIX%\share\foldtree2\mafft_tools
    if exist %PREFIX%\share\foldtree2\mafft_tools\hex2maffttext.exe (
        copy %PREFIX%\share\foldtree2\mafft_tools\hex2maffttext.exe %PREFIX%\Scripts\hex2maffttext.exe
    )
    if exist %PREFIX%\share\foldtree2\mafft_tools\maffttext2hex.exe (
        copy %PREFIX%\share\foldtree2\mafft_tools\maffttext2hex.exe %PREFIX%\Scripts\maffttext2hex.exe
    )
)

:: Copy configuration files
if exist foldtree2\config (
    xcopy /E /I foldtree2\config %PREFIX%\share\foldtree2\config
)

:: Always create production models directory; copy files when present
mkdir %PREFIX%\share\foldtree2\models\production 2>nul
if exist models\production (
    xcopy /E /I models\production %PREFIX%\share\foldtree2\models\production
)

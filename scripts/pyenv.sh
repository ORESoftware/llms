# 1. Install pyenv
curl https://pyenv.run | bash

# 2. Add pyenv to your shell startup file
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# 3. Install the desired Python version using pyenv
pyenv install 3.8.10

# 4. Set the local Python version for your project directory
pyenv local 3.8.10

# 5. Verify the Python version
python --version
# Output should be: Python 3.8.10

# 6. Create a virtual environment with venv
python -m venv myenv

# 7. Activate the virtual environment
# On Unix or MacOS
source myenv/bin/activate

# On Windows
myenv\Scripts\activate

# 8. Verify the Python version in the virtual environment
python --version
# Output should be: Python 3.8.10

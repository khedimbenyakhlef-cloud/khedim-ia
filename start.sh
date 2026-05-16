#!/bin/bash
echo "=== Patching Gradio Jinja2 bug ==="
python -c "
import gradio, os

# Patcher jinja2/utils.py - c'est la que le bug est
import jinja2
jinja_path = os.path.join(os.path.dirname(jinja2.__file__), 'utils.py')
content = open(jinja_path).read()

if 'khedim_patch' not in content:
    content = content.replace(
        'return self._mapping[key]',
        'return self._mapping[key if not isinstance(key, dict) else str(sorted(key.items()))]  # khedim_patch'
    )
    open(jinja_path, 'w').write(content)
    print('PATCH JINJA2 APPLIQUE')
else:
    print('Deja patche')
"
echo "=== Demarrage app ==="
python app.py

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the corrupted character
content = content.replace('with st.expander("� Dataset Configuration"):', 'with st.expander("📋 Dataset Configuration (data.yaml)"):')

# Also improve the YAML display to show as code instead of JSON
old_yaml_display = '''                        with st.expander("📋 Dataset Configuration (data.yaml)"):
                            try:
                                import yaml
                                with open(os.path.join(dataset_path, "data.yaml"), 'r') as f:
                                    config = yaml.safe_load(f)
                                    st.json(config)
                            except Exception as e:
                                st.error(f"Could not read data.yaml: {str(e)}")'''

new_yaml_display = '''                        with st.expander("📋 Dataset Configuration (data.yaml)"):
                            try:
                                with open(os.path.join(dataset_path, "data.yaml"), 'r') as f:
                                    yaml_content = f.read()
                                st.code(yaml_content, language="yaml")
                            except Exception as e:
                                st.error(f"Could not read data.yaml: {e}")'''

content = content.replace(old_yaml_display, new_yaml_display)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed corrupted character and improved YAML display')

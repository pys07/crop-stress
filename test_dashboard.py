#!/usr/bin/env python
from web_app import app

with app.test_client() as client:
    response = client.get('/dashboard')
    html = response.get_data(as_text=True)
    
    # Find the initCharts script section
    start = html.find('function initCharts')
    if start != -1:
        end = html.find('</script>', start)
        script = html[start:end]
        
        # Find the labels line
        labels_start = script.find('labels:')
        if labels_start != -1:
            labels_end = script.find(',', labels_start)
            print('LABELS LINE:')
            print(script[labels_start:labels_end + 50])
            print()
        
        # Find the data line  
        data_start = script.find('data:')
        if data_start != -1:
            data_end = script.find('}}', data_start) + 2
            print('DATA LINE:')
            print(script[data_start:min(data_end + 50, len(script))])

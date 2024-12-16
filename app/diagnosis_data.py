class DiagnosisData:
    FINDINGS_MAP = {
        'nil control': ['No significant pathological findings'],
        'condensing osteitis': ['Dense bone formation', 'Localized bone sclerosis'],
        'diffuse lesion': ['Widespread bone changes', 'Multiple affected areas'],
        'periapical abcess': ['Radiolucent area around root tip', 'Bone destruction'],
        'periapical granuloma': ['Small radiolucent lesion at root apex'],
        'periapical widening': ['Widened periodontal ligament space'],
        'pericoronitis': ['Soft tissue inflammation around partially erupted tooth'],
        'radicular cyst': ['Well-defined radiolucent lesion', 'Root resorption possible']
    }
    
    RECOMMENDATIONS_MAP = {
        'nil control': ['Regular dental check-ups'],
        'condensing osteitis': ['Endodontic evaluation', 'Monitor for changes'],
        'diffuse lesion': ['Comprehensive examination', 'Biopsy may be needed'],
        'periapical abcess': ['Immediate dental intervention', 'Root canal or extraction'],
        'periapical granuloma': ['Endodontic treatment', 'Regular monitoring'],
        'periapical widening': ['Dental evaluation', 'Possible endodontic treatment'],
        'pericoronitis': ['Oral hygiene instruction', 'Possible extraction'],
        'radicular cyst': ['Surgical evaluation', 'Endodontic treatment']
    }
    
    MANAGEMENT_MAP = {
        'nil control': {
            'immediate': ['Continue regular dental hygiene'],
            'long_term': ['Regular check-ups']
        },
        'condensing osteitis': {
            'immediate': ['Pain management if needed'],
            'long_term': ['Root canal treatment if symptomatic']
        },
        'diffuse lesion': {
            'immediate': ['Further diagnostic tests'],
            'long_term': ['Treatment based on diagnosis']
        },
        'periapical abcess': {
            'immediate': ['Antibiotics', 'Drainage if needed'],
            'long_term': ['Root canal or extraction']
        },
        'periapical granuloma': {
            'immediate': ['Pain management'],
            'long_term': ['Root canal treatment']
        },
        'periapical widening': {
            'immediate': ['Identify cause'],
            'long_term': ['Treat underlying condition']
        },
        'pericoronitis': {
            'immediate': ['Irrigation', 'Antibiotics if needed'],
            'long_term': ['Possible extraction']
        },
        'radicular cyst': {
            'immediate': ['Pain management'],
            'long_term': ['Surgical removal or root canal']
        }
    } 
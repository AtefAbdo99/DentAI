class DiagnosisData:
    FINDINGS_MAP = {
        'nil control': ['No pathological findings', 'Normal anatomical structures'],
        'condensing osteitis': [
            'Increased bone density around tooth apex',
            'Localized sclerotic bone response',
            'Associated with chronic low-grade pulpal inflammation'
        ],
        'diffuse lesion': [
            'Poorly defined radiolucent area',
            'Irregular borders',
            'Possible bone infiltration'
        ],
        'periapical abscess': [
            'Well-defined radiolucent area around tooth apex',
            'Possible loss of lamina dura',
            'Widening of periodontal ligament space'
        ],
        'periapical granuloma': [
            'Small well-defined radiolucent lesion',
            'Usually less than 1cm in diameter',
            'Associated with non-vital tooth'
        ],
        'periapical widening': [
            'Increased periodontal ligament space',
            'Continuous with lamina dura',
            'Early sign of periapical pathology'
        ],
        'pericoronitis': [
            'Soft tissue inflammation around partially erupted tooth',
            'Usually associated with third molars',
            'Possible bone loss pattern'
        ],
        'radicular cyst': [
            'Well-defined radiolucent lesion',
            'Usually larger than 1cm in diameter',
            'Sclerotic border present'
        ]
    }
    
    RECOMMENDATIONS_MAP = {
        'nil control': ['Regular dental check-ups', 'Maintain good oral hygiene'],
        'condensing osteitis': [
            'Endodontic evaluation recommended',
            'Monitor for changes in symptoms',
            'Consider vitality testing'
        ],
        'diffuse lesion': [
            'Immediate specialist referral',
            'Biopsy may be necessary',
            'Further imaging recommended'
        ],
        'periapical abscess': [
            'Urgent endodontic treatment required',
            'Possible antibiotic therapy',
            'Pain management as needed'
        ],
        'periapical granuloma': [
            'Root canal treatment indicated',
            'Regular follow-up required',
            'Monitor for healing progress'
        ],
        'periapical widening': [
            'Clinical correlation needed',
            'Vitality testing recommended',
            'Monitor for progression'
        ],
        'pericoronitis': [
            'Oral hygiene instruction',
            'Consider extraction if recurrent',
            'Chlorhexidine rinses recommended'
        ],
        'radicular cyst': [
            'Surgical evaluation needed',
            'Endodontic treatment required',
            'Regular radiographic follow-up'
        ]
    }
    
    MANAGEMENT_MAP = {
        'nil control': {
            'follow_up': '6-12 months routine check-up',
            'imaging': 'Routine radiographs as needed'
        },
        'condensing osteitis': {
            'primary': 'Endodontic treatment',
            'follow_up': '3-6 months',
            'imaging': 'Follow-up radiographs at 6 months'
        },
        'diffuse lesion': {
            'primary': 'Specialist referral',
            'follow_up': '2-4 weeks',
            'imaging': 'Advanced imaging (CBCT) recommended'
        },
        'periapical abscess': {
            'primary': 'Emergency endodontic treatment',
            'medication': 'Antibiotics if indicated',
            'follow_up': '1 week'
        },
        'periapical granuloma': {
            'primary': 'Root canal treatment',
            'follow_up': '3 months',
            'imaging': 'Follow-up radiographs at 6 months'
        },
        'periapical widening': {
            'primary': 'Clinical assessment',
            'follow_up': '2-4 weeks',
            'monitoring': 'Regular vitality testing'
        },
        'pericoronitis': {
            'primary': 'Local debridement',
            'medication': 'Chlorhexidine rinses',
            'follow_up': '1-2 weeks'
        },
        'radicular cyst': {
            'primary': 'Surgical enucleation',
            'secondary': 'Endodontic treatment',
            'follow_up': '3-6 months'
        }
    } 
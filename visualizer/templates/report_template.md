# {{ title }}

**Date:** {{ date }}

## Overview

This report presents the security analysis of the codebase, including identified threats, risk assessment, and recommendations.

## Risk Assessment

**Overall Risk Level:** {{ overall_risk }}

## Components Analysis

{% for component in components %}
### {{ component.name }} ({{ component.type }})

{% if component.threats %}
#### Identified Threats
{% for threat in component.threats %}
- **Type:** {{ threat.type }}
- **Severity:** {{ threat.severity }}
- **Description:** {{ threat.description }}
- **Impact:** {{ threat.impact }}
- **Mitigation:** {{ threat.mitigation }}
{% if threat.code_snippet %}
```python
{{ threat.code_snippet }}
```
{% endif %}
{% endfor %}
{% else %}
No threats identified.
{% endif %}
{% endfor %}

## Data Flow Analysis

{% for flow in data_flows %}
### {{ flow.source }} -> {{ flow.function }}

{% if flow.threats %}
#### Identified Threats
{% for threat in flow.threats %}
- **Type:** {{ threat.type }}
- **Severity:** {{ threat.severity }}
- **Description:** {{ threat.description }}
- **Impact:** {{ threat.impact }}
- **Mitigation:** {{ threat.mitigation }}
{% if threat.code_snippet %}
```python
{{ threat.code_snippet }}
```
{% endif %}
{% endfor %}
{% else %}
No threats identified.
{% endif %}
{% endfor %}

## Recommendations

{% for rec in recommendations %}
### {{ rec.type }}: {{ rec.target }}
- **Priority:** {{ rec.priority }}
- **Description:** {{ rec.description }}
- **Action:** {{ rec.action }}
{% endfor %} 
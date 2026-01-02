"""
Enterprise Detection Engine
Post-AI Rule Engine with Tri-State Classification, Time Correlation, and Explainability

This transforms raw ML predictions into enterprise-grade threat intelligence.
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ThreatLevel:
    """Tri-state threat classification"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious" 
    ATTACK = "attack"


class AlertSeverity:
    """Alert severity levels for SOC integration"""
    IGNORE = "ignore"
    LOG = "log"
    MONITOR = "monitor"
    ALERT = "alert"
    BLOCK = "block"


class DetectionEngine:
    """
    Enterprise-grade detection engine that wraps ML predictions
    with rule-based logic, time correlation, and explainability.
    """
    
    # Feature names for explainability
    FEATURE_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    # Comprehensive attack indicators - supports 25+ vulnerability types
    ATTACK_INDICATORS = {
        # DoS Attack Indicators
        'serror_rate': {
            'threshold': 0.5, 
            'name': 'SYN Flood Attack', 
            'description': 'SYN flood detected - half-open TCP connections exhausting resources',
            'category': 'DoS',
            'severity': 'CRITICAL',
            'cve': 'CVE-1999-0116'
        },
        'srv_serror_rate': {
            'threshold': 0.5, 
            'name': 'Service SYN Flood', 
            'description': 'Targeted SYN flood on specific service port',
            'category': 'DoS',
            'severity': 'HIGH'
        },
        'rerror_rate': {
            'threshold': 0.5, 
            'name': 'RST Flood Attack', 
            'description': 'High rate of rejected connections - possible RST flood',
            'category': 'DoS',
            'severity': 'HIGH'
        },
        'dst_host_serror_rate': {
            'threshold': 0.5, 
            'name': 'Host-based DoS', 
            'description': 'Destination host experiencing SYN flood attack',
            'category': 'DoS',
            'severity': 'CRITICAL'
        },
        'dst_host_rerror_rate': {
            'threshold': 0.5, 
            'name': 'Connection Exhaustion', 
            'description': 'High rejection rate - server resource exhaustion',
            'category': 'DoS',
            'severity': 'HIGH'
        },
        
        # Probe/Reconnaissance Attack Indicators
        'count': {
            'threshold': 100, 
            'name': 'Port Scan Detected', 
            'description': 'Excessive connections - likely port scanning/reconnaissance',
            'category': 'Probe',
            'severity': 'MEDIUM',
            'tool': 'Nmap/Masscan'
        },
        'srv_count': {
            'threshold': 100, 
            'name': 'Service Enumeration', 
            'description': 'High service requests - possible service enumeration',
            'category': 'Probe',
            'severity': 'MEDIUM'
        },
        'diff_srv_rate': {
            'threshold': 0.5, 
            'name': 'Multi-Service Probe', 
            'description': 'Connections to many different services - reconnaissance',
            'category': 'Probe',
            'severity': 'MEDIUM'
        },
        'dst_host_diff_srv_rate': {
            'threshold': 0.5, 
            'name': 'Host Service Scanning', 
            'description': 'Scanning multiple services on target host',
            'category': 'Probe',
            'severity': 'MEDIUM'
        },
        
        # Brute Force / Credential Attack Indicators
        'num_failed_logins': {
            'threshold': 3, 
            'name': 'Brute Force Attack', 
            'description': 'Multiple failed login attempts - password guessing/brute force',
            'category': 'R2L',
            'severity': 'HIGH',
            'mitre': 'T1110'
        },
        'su_attempted': {
            'threshold': 1, 
            'name': 'Privilege Escalation Attempt', 
            'description': 'Attempted to switch user - possible privilege escalation',
            'category': 'U2R',
            'severity': 'CRITICAL',
            'mitre': 'T1548'
        },
        
        # Exploitation Indicators
        'root_shell': {
            'threshold': 1, 
            'name': 'Root Shell Access', 
            'description': 'Root shell obtained - system fully compromised!',
            'category': 'U2R',
            'severity': 'CRITICAL',
            'mitre': 'T1059'
        },
        'num_shells': {
            'threshold': 1, 
            'name': 'Shell Spawned', 
            'description': 'Command shell spawned - possible reverse shell',
            'category': 'U2R',
            'severity': 'CRITICAL'
        },
        'num_compromised': {
            'threshold': 1, 
            'name': 'Compromise Indicator', 
            'description': 'System shows signs of compromise',
            'category': 'U2R',
            'severity': 'CRITICAL'
        },
        'num_root': {
            'threshold': 1, 
            'name': 'Root Access Operations', 
            'description': 'Root-level operations detected',
            'category': 'U2R',
            'severity': 'CRITICAL'
        },
        
        # Data Exfiltration Indicators
        'num_file_creations': {
            'threshold': 5, 
            'name': 'Suspicious File Activity', 
            'description': 'Multiple file creations - possible data staging',
            'category': 'Exfiltration',
            'severity': 'MEDIUM',
            'mitre': 'T1074'
        },
        'num_access_files': {
            'threshold': 5, 
            'name': 'File Access Anomaly', 
            'description': 'Unusual file access pattern - data theft possible',
            'category': 'Exfiltration',
            'severity': 'MEDIUM'
        },
        'num_outbound_cmds': {
            'threshold': 1, 
            'name': 'Outbound Commands', 
            'description': 'Outbound command execution - C2 communication',
            'category': 'C2',
            'severity': 'HIGH',
            'mitre': 'T1071'
        },
        
        # Network Anomaly Indicators
        'land': {
            'threshold': 1, 
            'name': 'Land Attack', 
            'description': 'Source and destination IP/port match - Land attack detected',
            'category': 'DoS',
            'severity': 'HIGH',
            'cve': 'CVE-1997-0327'
        },
        'wrong_fragment': {
            'threshold': 1, 
            'name': 'Fragment Attack', 
            'description': 'Malformed packet fragments - evasion or exploit attempt',
            'category': 'Exploit',
            'severity': 'HIGH'
        },
        'urgent': {
            'threshold': 1, 
            'name': 'Urgent Pointer Abuse', 
            'description': 'Suspicious use of TCP urgent pointer',
            'category': 'Exploit',
            'severity': 'MEDIUM'
        },
        'hot': {
            'threshold': 5, 
            'name': 'Hot Indicators', 
            'description': 'Suspicious system commands detected',
            'category': 'Suspicious',
            'severity': 'MEDIUM'
        },
        'is_guest_login': {
            'threshold': 1, 
            'name': 'Guest Login', 
            'description': 'Guest account used - potential unauthorized access',
            'category': 'R2L',
            'severity': 'LOW'
        },
    }
    
    # Normal traffic indicators
    NORMAL_INDICATORS = {
        'logged_in': {'value': 1, 'name': 'Authenticated Session', 'description': 'Valid authenticated session'},
        'same_srv_rate': {'threshold': 0.9, 'name': 'Consistent Service', 'description': 'Normal service usage pattern'},
        'dst_host_same_srv_rate': {'threshold': 0.9, 'name': 'Normal Host Pattern', 'description': 'Typical host access pattern'},
    }
    
    # Attack category descriptions for user-friendly output
    ATTACK_CATEGORIES = {
        'DoS': {
            'name': 'Denial of Service',
            'icon': '💥',
            'description': 'Attacks designed to make services unavailable',
            'recommendation': 'Enable rate limiting, deploy DDoS protection, block source IP'
        },
        'Probe': {
            'name': 'Reconnaissance',
            'icon': '🔍',
            'description': 'Network scanning and information gathering',
            'recommendation': 'Review firewall rules, implement port knocking, monitor for follow-up attacks'
        },
        'R2L': {
            'name': 'Remote to Local',
            'icon': '🔓',
            'description': 'Unauthorized remote access attempts',
            'recommendation': 'Enable MFA, implement account lockout, review access logs'
        },
        'U2R': {
            'name': 'User to Root (Privilege Escalation)',
            'icon': '👤',
            'description': 'Attempts to gain elevated privileges',
            'recommendation': 'IMMEDIATE ACTION: Isolate system, review sudo/SUID, run rootkit scan'
        },
        'Exfiltration': {
            'name': 'Data Exfiltration',
            'icon': '📤',
            'description': 'Potential data theft detected',
            'recommendation': 'Monitor outbound traffic, review file access, enable DLP'
        },
        'C2': {
            'name': 'Command & Control',
            'icon': '📡',
            'description': 'Suspected malware communication',
            'recommendation': 'Block outbound traffic, isolate host, run malware scan'
        },
        'Exploit': {
            'name': 'Exploitation Attempt',
            'icon': '💀',
            'description': 'Active exploitation detected',
            'recommendation': 'Patch systems, review WAF logs, check for compromised services'
        },
        'Suspicious': {
            'name': 'Suspicious Activity',
            'icon': '⚠️',
            'description': 'Unusual behavior requiring investigation',
            'recommendation': 'Add to watchlist, correlate with other logs, investigate'
        }
    }
    
    def __init__(self):
        # Time-based correlation: track events by source IP
        self.ip_history: Dict[str, List[Dict]] = defaultdict(list)
        self.history_window = timedelta(minutes=5)  # 5-minute sliding window
        
        # Statistics
        self.total_analyzed = 0
        self.threats_detected = 0
        self.suspicious_count = 0
        
    def analyze(
        self, 
        ml_prediction: int, 
        ml_confidence: float, 
        features: Dict[str, Any],
        source_ip: Optional[str] = None,
        ml_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main analysis function - combines ML prediction with rule engine.
        
        Args:
            ml_prediction: 0 = Normal, 1 = Attack (binary for compatibility)
            ml_confidence: Confidence score 0-1
            features: Raw feature dictionary
            source_ip: Source IP for correlation (optional)
            ml_label: The actual label from ML model (e.g., 'Normal', 'Neptune', 'Smurf')
            
        Returns:
            Comprehensive analysis result
        """
        self.total_analyzed += 1
        timestamp = datetime.now()
        
        # 1. Get rule-based assessment
        rule_flags, rule_score = self._apply_rules(features)
        
        # 2. Determine threat level using tri-state classification
        threat_level, final_confidence = self._classify_threat(
            ml_prediction, ml_confidence, rule_score, rule_flags, ml_label
        )
        
        # 3. Time-based correlation (if IP provided)
        correlation_result = None
        if source_ip:
            correlation_result = self._correlate_events(source_ip, threat_level, timestamp)
            # Escalate if repeated suspicious activity
            if correlation_result['escalate']:
                threat_level = ThreatLevel.ATTACK
                final_confidence = max(final_confidence, 0.85)
        
        # 4. Traffic direction analysis
        direction_analysis = self._analyze_traffic_direction(features)
        
        # 5. Generate explanation
        explanation = self._generate_explanation(
            ml_prediction, ml_confidence, threat_level, 
            rule_flags, features, direction_analysis
        )
        
        # 6. Determine action and severity
        action, severity = self._determine_action(threat_level, final_confidence)
        
        # Update statistics
        if threat_level == ThreatLevel.ATTACK:
            self.threats_detected += 1
        elif threat_level == ThreatLevel.SUSPICIOUS:
            self.suspicious_count += 1
        
        return {
            'timestamp': timestamp.isoformat(),
            'threat_level': threat_level,
            'confidence': round(final_confidence * 100, 1),
            'ml_prediction': 'Attack' if ml_prediction == 1 else 'Normal',
            'ml_confidence': round(ml_confidence * 100, 1),
            'action': action,
            'severity': severity,
            'explanation': explanation,
            'rule_flags': rule_flags,
            'direction_analysis': direction_analysis,
            'correlation': correlation_result,
            'top_factors': self._get_top_factors(rule_flags, features),
            'stats': {
                'total_analyzed': self.total_analyzed,
                'threats_detected': self.threats_detected,
                'suspicious_count': self.suspicious_count
            }
        }
    
    def _apply_rules(self, features: Dict[str, Any]) -> Tuple[List[Dict], float]:
        """
        Apply rule-based detection to supplement ML.
        Returns list of triggered flags and overall rule score.
        """
        flags = []
        score = 0.0
        
        # Check attack indicators
        for key, config in self.ATTACK_INDICATORS.items():
            if key in features:
                value = features.get(key, 0)
                try:
                    value = float(value)
                    if value >= config['threshold']:
                        severity = min(value / config['threshold'], 2.0)
                        flags.append({
                            'indicator': config['name'],
                            'description': config['description'],
                            'value': value,
                            'threshold': config['threshold'],
                            'type': 'attack',
                            'severity': severity
                        })
                        score += severity * 0.15
                except (ValueError, TypeError):
                    pass
        
        # Check normal indicators (reduce score if present)
        for key, config in self.NORMAL_INDICATORS.items():
            if key in features:
                value = features.get(key, 0)
                try:
                    value = float(value)
                    threshold = config.get('threshold', config.get('value', 0))
                    if value >= threshold:
                        flags.append({
                            'indicator': config['name'],
                            'description': config['description'],
                            'value': value,
                            'type': 'normal'
                        })
                        score -= 0.1
                except (ValueError, TypeError):
                    pass
        
        # Special pattern detection
        
        # Pattern 1: SYN flood detection (S0 flag with high count)
        flag_val = features.get('flag', '')
        count_val = float(features.get('count', 0))
        if flag_val == 'S0' and count_val > 50:
            flags.append({
                'indicator': 'SYN Flood Pattern',
                'description': 'High count of half-open connections (S0 flag)',
                'type': 'attack',
                'severity': 1.5
            })
            score += 0.3
        
        # Pattern 2: Port scan detection (REJ flag with high count)
        if flag_val == 'REJ' and count_val > 100:
            flags.append({
                'indicator': 'Port Scan Pattern',
                'description': 'Many rejected connection attempts',
                'type': 'attack',
                'severity': 1.2
            })
            score += 0.25
        
        return flags, max(0, min(score, 1.0))
    
    def _classify_threat(
        self, 
        ml_pred: int, 
        ml_conf: float, 
        rule_score: float,
        rule_flags: List[Dict],
        ml_label: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Tri-state classification: Normal / Suspicious / Attack
        
        Uses ml_label if provided (for 23-class model), otherwise uses ml_pred.
        With binary ml_pred: 0 = Normal, 1 = Attack
        """
        attack_flags = [f for f in rule_flags if f.get('type') == 'attack']
        
        # Combine ML confidence with rule score
        combined_score = (ml_conf * 0.7) + (rule_score * 0.3)
        
        # Determine if ML says Normal - use label if available, else use prediction
        is_ml_normal = ml_label == 'Normal' if ml_label else (ml_pred == 0)
        
        if not is_ml_normal:  # ML says attack
            if ml_conf >= 0.85 or len(attack_flags) >= 2:
                return ThreatLevel.ATTACK, combined_score
            elif ml_conf >= 0.6:
                return ThreatLevel.SUSPICIOUS, combined_score
            else:
                return ThreatLevel.SUSPICIOUS, combined_score * 0.8
        else:  # ML says Normal
            # Trust ML when it has high confidence, even if rules trigger
            if ml_conf >= 0.9:
                return ThreatLevel.SAFE, ml_conf
            elif ml_conf >= 0.75 and len(attack_flags) < 3:
                return ThreatLevel.SAFE, ml_conf
            elif len(attack_flags) >= 4:
                return ThreatLevel.SUSPICIOUS, 0.5 + (rule_score * 0.3)
            else:
                return ThreatLevel.SAFE, ml_conf
    
    def _correlate_events(
        self, 
        source_ip: str, 
        threat_level: str, 
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Time-based correlation: track events by IP to detect patterns.
        """
        # Skip correlation for localhost (demo/testing)
        if source_ip in ['127.0.0.1', 'localhost', '::1', None, '']:
            return {
                'ip': source_ip,
                'events_in_window': 0,
                'suspicious_events': 0,
                'escalate': False,
                'reason': None
            }
        
        # Clean old events
        cutoff = timestamp - self.history_window
        self.ip_history[source_ip] = [
            e for e in self.ip_history[source_ip] 
            if e['timestamp'] > cutoff
        ]
        
        # Add current event
        self.ip_history[source_ip].append({
            'timestamp': timestamp,
            'threat_level': threat_level
        })
        
        events = self.ip_history[source_ip]
        suspicious_count = sum(1 for e in events if e['threat_level'] in [ThreatLevel.SUSPICIOUS, ThreatLevel.ATTACK])
        
        # Escalation rules (more conservative)
        escalate = False
        reason = None
        
        if len(events) >= 10 and suspicious_count >= 7:
            escalate = True
            reason = f"Repeated suspicious activity: {suspicious_count} events in {len(events)} connections"
        elif suspicious_count >= 10:
            escalate = True
            reason = f"High threat frequency: {suspicious_count} suspicious events in 5 minutes"
        
        return {
            'ip': source_ip,
            'events_in_window': len(events),
            'suspicious_events': suspicious_count,
            'escalate': escalate,
            'reason': reason
        }
    
    def _analyze_traffic_direction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze traffic direction for exfiltration detection.
        """
        src_bytes = float(features.get('src_bytes', 0))
        dst_bytes = float(features.get('dst_bytes', 0))
        
        # Calculate ratio
        if src_bytes > 0:
            out_in_ratio = dst_bytes / src_bytes if src_bytes > 0 else 0
        else:
            out_in_ratio = dst_bytes if dst_bytes > 0 else 0
        
        # Detect anomalies
        exfiltration_risk = False
        direction_note = "Normal bidirectional traffic"
        
        if dst_bytes > 50000 and out_in_ratio > 10:
            exfiltration_risk = True
            direction_note = "⚠️ Possible data exfiltration: Large outbound data transfer"
        elif src_bytes == 0 and dst_bytes > 10000:
            exfiltration_risk = True
            direction_note = "⚠️ One-way outbound traffic detected"
        elif dst_bytes == 0 and src_bytes > 0:
            direction_note = "Inbound-only traffic (normal for requests)"
        
        return {
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'ratio': round(out_in_ratio, 2),
            'exfiltration_risk': exfiltration_risk,
            'note': direction_note
        }
    
    def _generate_explanation(
        self, 
        ml_pred: int,
        ml_conf: float,
        threat_level: str,
        rule_flags: List[Dict],
        features: Dict[str, Any],
        direction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanation for the decision.
        """
        attack_flags = [f for f in rule_flags if f.get('type') == 'attack']
        normal_flags = [f for f in rule_flags if f.get('type') == 'normal']
        
        # Main explanation
        if threat_level == ThreatLevel.ATTACK:
            summary = "🚨 THREAT DETECTED: This traffic exhibits clear attack characteristics."
            recommendation = "Immediate investigation required. Consider blocking source IP."
        elif threat_level == ThreatLevel.SUSPICIOUS:
            summary = "⚠️ SUSPICIOUS ACTIVITY: This traffic shows concerning patterns that warrant monitoring."
            recommendation = "Add to watchlist. Monitor for repeated behavior."
        else:
            summary = "✅ SAFE: This traffic appears to be legitimate network activity."
            recommendation = "No action required."
        
        # Build detailed reasoning
        reasons = []
        
        # ML reasoning
        ml_label = "Attack" if ml_pred == 1 else "Normal"
        reasons.append(f"AI Model classified as {ml_label} with {ml_conf*100:.1f}% confidence")
        
        # Rule reasoning
        if attack_flags:
            reasons.append(f"Rule engine detected {len(attack_flags)} attack indicator(s)")
            for f in attack_flags[:3]:  # Top 3
                reasons.append(f"  • {f['indicator']}: {f['description']}")
        
        if normal_flags:
            reasons.append(f"Found {len(normal_flags)} normal traffic indicator(s)")
        
        # Direction reasoning
        if direction['exfiltration_risk']:
            reasons.append(f"Traffic direction alert: {direction['note']}")
        
        return {
            'summary': summary,
            'recommendation': recommendation,
            'reasons': reasons,
            'attack_indicators': len(attack_flags),
            'normal_indicators': len(normal_flags)
        }
    
    def _get_top_factors(
        self, 
        rule_flags: List[Dict], 
        features: Dict[str, Any]
    ) -> List[Dict]:
        """
        Get top contributing factors for the decision.
        """
        factors = []
        
        # Add rule-based factors
        for flag in rule_flags:
            if flag.get('type') == 'attack':
                factors.append({
                    'name': flag['indicator'],
                    'impact': 'HIGH',
                    'value': flag.get('value', 'N/A'),
                    'description': flag['description']
                })
        
        # Add key feature values
        key_features = ['count', 'srv_count', 'serror_rate', 'duration', 'src_bytes', 'dst_bytes']
        for feat in key_features:
            if feat in features and len(factors) < 5:
                val = features.get(feat, 0)
                factors.append({
                    'name': feat.replace('_', ' ').title(),
                    'impact': 'INFO',
                    'value': val,
                    'description': f'Feature value: {val}'
                })
        
        return factors[:5]  # Top 5
    
    def _determine_action(
        self, 
        threat_level: str, 
        confidence: float
    ) -> Tuple[str, str]:
        """
        Determine recommended action and severity.
        """
        if threat_level == ThreatLevel.ATTACK:
            if confidence >= 0.9:
                return "BLOCK", AlertSeverity.BLOCK
            else:
                return "ALERT", AlertSeverity.ALERT
        elif threat_level == ThreatLevel.SUSPICIOUS:
            if confidence >= 0.7:
                return "MONITOR", AlertSeverity.MONITOR
            else:
                return "LOG", AlertSeverity.LOG
        else:
            return "ALLOW", AlertSeverity.IGNORE


# Singleton instance
_engine = None

def get_engine() -> DetectionEngine:
    """Get or create the detection engine singleton."""
    global _engine
    if _engine is None:
        _engine = DetectionEngine()
    return _engine

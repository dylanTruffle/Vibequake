#!/usr/bin/env python3
import subprocess
import requests
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def check_domain(domain):
    """Check if a domain resolves and returns a response"""
    try:
        # Try to resolve the domain
        socket.gethostbyname(domain)
        
        # Try HTTP request
        try:
            response = requests.get(f"https://{domain}", timeout=5, allow_redirects=True)
            if response.status_code == 200:
                print(f"✅ FOUND: {domain} (Status: {response.status_code})")
                print(f"   Content length: {len(response.text)}")
                if "paywall" in response.text.lower() or "vpn" in response.text.lower():
                    print(f"   ⚠️  Contains paywall/VPN keywords!")
                return domain
        except requests.exceptions.RequestException:
            pass
            
        # Try HTTP
        try:
            response = requests.get(f"http://{domain}", timeout=5, allow_redirects=True)
            if response.status_code == 200:
                print(f"✅ FOUND: {domain} (HTTP Status: {response.status_code})")
                print(f"   Content length: {len(response.text)}")
                if "paywall" in response.text.lower() or "vpn" in response.text.lower():
                    print(f"   ⚠️  Contains paywall/VPN keywords!")
                return domain
        except requests.exceptions.RequestException:
            pass
            
        print(f"✅ Resolves: {domain}")
        return domain
        
    except socket.gaierror:
        return None

def main():
    base = "paywalloptional"
    
    # Common TLDs
    tlds = [
        "com", "net", "org", "io", "dev", "tech", "app", "site", "xyz", "cloud",
        "aws", "ctf", "hack", "pw", "me", "co", "info", "biz", "us", "uk",
        "de", "fr", "it", "es", "nl", "ca", "au", "jp", "cn", "in", "ru",
        "br", "mx", "ar", "cl", "pe", "co.za", "ng", "ke", "gh", "ug", "tz",
        "zw", "na", "bw", "mw", "zm", "ao", "mz", "st", "sc", "mu", "re",
        "yt", "dj", "km", "so", "et", "sd", "ss", "cf", "cg", "cd", "gq",
        "ga", "td", "ne", "ml"
    ]
    
    # Subdomain patterns
    subdomains = [
        "www", "api", "portal", "vpn", "corp", "internal", "admin", "secure",
        "access", "gateway", "proxy", "tunnel", "bridge", "connect", "login",
        "auth", "sso", "ldap", "radius", "vpn1", "vpn2", "corp1", "corp2",
        "internal1", "internal2", "admin1", "admin2", "secure1", "secure2",
        "portal1", "portal2", "api1", "api2", "www1", "www2", "gateway1",
        "gateway2", "proxy1", "proxy2", "tunnel1", "tunnel2", "bridge1",
        "bridge2", "connect1", "connect2", "login1", "login2", "auth1",
        "auth2", "sso1", "sso2", "ldap1", "ldap2", "radius1", "radius2"
    ]
    
    # Split patterns (with dots)
    splits = [
        "paywall.optional", "pay.walloptional", "paywallopt.ional",
        "pay.wall.optional", "paywall.opt.ional", "pay.wall.opt.ional",
        "p.aywalloptional", "pa.ywalloptional", "pay.walloptional",
        "payw.alloptional", "paywa.lloptional", "paywal.loptional",
        "paywall.optional", "paywallo.ptional", "paywallop.tional",
        "paywallopt.ional", "paywallopti.onal", "paywalloptio.nal",
        "paywalloption.al", "paywalloptiona.l"
    ]
    
    # AWS-specific patterns
    aws_patterns = [
        "paywalloptional.aws", "paywalloptional.amazonaws.com",
        "paywalloptional.s3.amazonaws.com", "paywalloptional.cloudfront.net",
        "paywalloptional.elasticbeanstalk.com", "paywalloptional.ec2.amazonaws.com",
        "paywalloptional.elb.amazonaws.com", "paywalloptional.rds.amazonaws.com",
        "paywalloptional.lambda.amazonaws.com", "paywalloptional.api-gateway.amazonaws.com"
    ]
    
    # CTF-specific patterns
    ctf_patterns = [
        "paywalloptional.ctf", "ctf.paywalloptional", "hack.paywalloptional",
        "challenge.paywalloptional", "flag.paywalloptional", "secret.paywalloptional",
        "hidden.paywalloptional", "internal.paywalloptional", "corp.paywalloptional",
        "vpn.paywalloptional", "portal.paywalloptional", "gateway.paywalloptional"
    ]
    
    all_domains = []
    
    # Add basic TLD variations
    for tld in tlds:
        all_domains.append(f"{base}.{tld}")
    
    # Add subdomain variations
    for subdomain in subdomains:
        for tld in ["com", "net", "org", "io", "dev", "tech", "app", "site", "xyz"]:
            all_domains.append(f"{subdomain}.{base}.{tld}")
            all_domains.append(f"{base}.{subdomain}.{tld}")
    
    # Add split variations
    for split in splits:
        for tld in ["com", "net", "org", "io", "dev", "tech", "app", "site", "xyz"]:
            all_domains.append(f"{split}.{tld}")
    
    # Add AWS patterns
    all_domains.extend(aws_patterns)
    
    # Add CTF patterns
    all_domains.extend(ctf_patterns)
    
    # Add some additional variations
    variations = [
        "paywall-optional", "paywall_optional", "paywalloptional-ctf",
        "paywalloptional-hack", "paywalloptional-challenge", "paywalloptional-secret",
        "paywalloptional-hidden", "paywalloptional-internal", "paywalloptional-corp",
        "paywalloptional-vpn", "paywalloptional-portal", "paywalloptional-gateway"
    ]
    
    for variation in variations:
        for tld in ["com", "net", "org", "io", "dev", "tech", "app", "site", "xyz"]:
            all_domains.append(f"{variation}.{tld}")
    
    print(f"🔍 Starting reconnaissance on {len(all_domains)} domains...")
    print("=" * 60)
    
    found_domains = []
    
    # Use ThreadPoolExecutor for concurrent checking
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_domain, domain): domain for domain in all_domains}
        
        for future in futures:
            result = future.result()
            if result:
                found_domains.append(result)
    
    print("\n" + "=" * 60)
    print(f"🎯 Reconnaissance complete!")
    print(f"📊 Checked {len(all_domains)} domains")
    print(f"✅ Found {len(found_domains)} resolving domains:")
    
    for domain in found_domains:
        print(f"   • {domain}")
    
    if found_domains:
        print(f"\n🔍 Checking found domains for paywall/VPN content...")
        for domain in found_domains:
            try:
                response = requests.get(f"https://{domain}", timeout=10)
                content = response.text.lower()
                if "paywall" in content or "vpn" in content or "corporate" in content:
                    print(f"🎯 POTENTIAL TARGET: {domain}")
                    print(f"   Status: {response.status_code}")
                    print(f"   Content preview: {response.text[:200]}...")
            except:
                try:
                    response = requests.get(f"http://{domain}", timeout=10)
                    content = response.text.lower()
                    if "paywall" in content or "vpn" in content or "corporate" in content:
                        print(f"🎯 POTENTIAL TARGET: {domain}")
                        print(f"   Status: {response.status_code}")
                        print(f"   Content preview: {response.text[:200]}...")
                except:
                    pass

if __name__ == "__main__":
    main()
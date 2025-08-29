# Date updated: 2024-02-19
# Author: Patrick Major
# Description: This script exports self-signed certificates from the LocalMachine\My store to a file in PEM format.
# Usage: ExportSelfSignedCertificate.ps1 -certOutputPath "C:\certs\self-signed.crt"

param (
    [string]$certOutputPath,
    [switch]$exportAllCerts
)

function Format-Base64Certificate {
    param (
        [string]$base64String
    )

    $lines = for ($i = 0; $i -lt $base64String.Length; $i += 64) {
        if ($i + 64 -lt $base64String.Length) {
            $base64String.Substring($i, 64)
        } else {
            $base64String.Substring($i)
        }
    }

    return $lines -join "`n"
}

function Export-Certificates {
    param (
        [string]$certOutputPath="C:\certs\ca-bundle.trust.crt",
        [switch]$exportAllCerts=$true
    )

    # Initialize array to store certificates
    $certsToExport = @()

    # Get certificates from different stores
    $stores = @(
        "Cert:\LocalMachine\Root",
        "Cert:\LocalMachine\CA",
        "Cert:\LocalMachine\My"
    )

    foreach ($storePath in $stores) {
        if ($exportAllCerts) {
            $storeCerts = Get-ChildItem -Path $storePath
            $certsToExport += $storeCerts
        } else {
            $storeCerts = Get-ChildItem -Path $storePath | Where-Object { 
                $_.Subject -like "*WSBC*" -or 
                $_.Subject -like "*inspect.worksafebc.com*" -or
                $_.DnsNameList -contains "inspect.worksafebc.com"
            }
            $certsToExport += $storeCerts
        }
    }

    # Remove duplicates based on thumbprint
    $certsToExport = $certsToExport | Sort-Object Thumbprint -Unique

    if (-not $certsToExport) {
        Write-Host "No certificates found matching the criteria."
        exit 1
    }

    # Create output directory if needed
    $outputFolder = [System.IO.Path]::GetDirectoryName($certOutputPath)
    if (-not (Test-Path -Path $outputFolder)) {
        New-Item -ItemType Directory -Path $outputFolder -Force | Out-Null
    }

    # Build bundle content
    $bundleContent = New-Object System.Collections.ArrayList

    foreach ($cert in $certsToExport) {
        if ($bundleContent.Count -gt 0) {
            $bundleContent.Add("") | Out-Null
        }

        # Add certificate information as comments
        $bundleContent.Add("# Issuer: $($cert.Issuer)") | Out-Null
        $bundleContent.Add("# Serial Number: $($cert.SerialNumber)") | Out-Null
        $bundleContent.Add("# Subject: $($cert.Subject)") | Out-Null
        $bundleContent.Add("# Not Valid Before: $($cert.NotBefore)") | Out-Null
        $bundleContent.Add("# Not Valid After: $($cert.NotAfter)") | Out-Null
        $bundleContent.Add("# Thumbprint: $($cert.Thumbprint)") | Out-Null

        if ($cert.DnsNameList) {
            $bundleContent.Add("# DNS Names: $($cert.DnsNameList -join ', ')") | Out-Null
        }

        # Add the PEM certificate
        $certData = [System.Convert]::ToBase64String($cert.RawData)
        $formattedCertData = Format-Base64Certificate -base64String $certData
        $bundleContent.Add("-----BEGIN CERTIFICATE-----") | Out-Null
        $bundleContent.Add($formattedCertData) | Out-Null
        $bundleContent.Add("-----END CERTIFICATE-----") | Out-Null
    }

    # Write to file (overwrite if exists)
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($certOutputPath, $bundleContent, $utf8NoBom)

    Write-Host "Exported $($certsToExport.Count) certificates to $certOutputPath"

    # Display certificate info
    Write-Host "`nIncluded certificates:"
    $certsToExport | Format-Table Subject, Thumbprint, NotAfter -AutoSize
}

Export-Certificates @PSBoundParameters
 
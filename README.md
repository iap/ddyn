# ns1-dhcpd-ddns-sync
Tool to synchronize NS1 DNS record management with an ISC DHCPD server

## Installation
Create an API key in the NS1 portal if you have not already done so by going to **Account** -> **Settings & Users** -> **Manage API Keys** -> **Add a New Key**.

Create a zone to contain your forward `A` records (`example.com`) and a zone to contain your reverse `PTR` records (`in-addr.arpa`) in the NS1 portal if you have not already done so by going to **Zones** -> **Add a New Zone**.

Copy the `ns1-dhcpd-ddns-sync.pl` script to a suitable location on your DHCP server, such as `/usr/local/bin`.

Add it to `root`'s crontab so it executes frequently, ideally every minute:

`* * * * * /usr/local/bin/ns1-dhcpd-ddns-sync.pl --api-key=<key> --forward-zone=<zone> --reverse-zone=<zone>`

`ns1-dhcpd-ddns-sync.pl` assumes that the DHCP server keeps track of active leases in the file `/var/lib/dhcpd/dhcpd.leases`.  If this is not the case, then you can tell `ns1-dhcpd-ddns-sync.pl` where to find the DHCP leases file with the optional `--leases-file=<file>` parameter.

## Usage
If you need to troubleshoot the synchronization process between your DHCP server and NS1, you can manually run `ns1-dhcpd-ddns-sync.pl` with the ``--verbose`` parameter, like so:

`ns1-dhcpd-ddns-sync.pl --verbose --api-key=<key> --forward-zone=<zone> --reverse-zone=<zone>`

In order to avoid clobbering other records within the forward and reverse zones, `ns1-dhcpd-ddns-sync.pl` stores the ethernet (MAC) address of the DHCP client in the **Notes** field for each corresponding DNS record.  When a lease expires, the DHCP client's ethernet address is removed from `/var/lib/dhcpd/dhcpd.leases`.  Any records found in the forward and reverse zones that contain an ethernet address that is not found in the current `/var/lib/dhcpd/dhcpd.leases` are removed, as the lease is assumed to have expired.  Records not containing an ethernet address in the **Notes** field are left intact.

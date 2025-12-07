def generate_simple_pdf(filename="presentation_all_in.pdf", num_pages=5):
    """
    Generates a very simple PDF with specified number of pages.
    Each page contains a title "Slide X".
    """
    pdf_content = []
    
    # PDF Header
    pdf_content.append("%PDF-1.4")
    
    # Object 1: Page tree root
    obj_pages_ref = "1 0 R"
    pdf_content.append("1 0 obj")
    pdf_content.append("<< /Type /Pages")
    pdf_content.append(f"/Count {num_pages}")
    # Placeholder for /Kids, will be filled later
    pdf_content.append("/Kids [] >>") 
    pdf_content.append("endobj")
    
    # Page objects and their content streams
    page_refs = []
    content_stream_refs = []
    current_obj_id = 2
    
    for i in range(num_pages):
        page_ref = f"{current_obj_id} 0 R"
        page_refs.append(page_ref)
        current_obj_id += 1
        
        content_stream_ref = f"{current_obj_id} 0 R"
        content_stream_refs.append(content_stream_ref)
        current_obj_id += 1

        # Content Stream Object
        text_content = f"BT /F1 24 Tf 100 700 Td ({i+1}. Slide Title) Tj ET"
        stream_length = len(text_content.encode('latin1')) # PDf requires length in bytes. 
        
        pdf_content.append(f"{content_stream_refs[-1].split(' ')[0]} 0 obj")
        pdf_content.append(f"<< /Length {stream_length} >>")
        pdf_content.append("stream")
        pdf_content.append(text_content)
        pdf_content.append("endstream")
        pdf_content.append("endobj")
        
        # Page Object
        pdf_content.append(f"{page_refs[-1].split(' ')[0]} 0 obj")
        pdf_content.append("<< /Type /Page")
        pdf_content.append(f"/Parent {obj_pages_ref}")
        pdf_content.append("/MediaBox [0 0 800 800]") # A custom box
        pdf_content.append("/Resources << /ProcSet [/PDF /Text] /Font << /F1 << /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica /Encoding /MacRomanEncoding >> >> >>")
        pdf_content.append(f"/Contents {content_stream_refs[-1]} >>")
        pdf_content.append("endobj")
        
    # Update Page tree root with actual page references
    kids_array_str = " ".join(page_refs)
    pdf_content[pdf_content.index(f"/Kids [] >>")] = f"/Kids [{kids_array_str}] >>"
    
    # Object for Catalog (Root)
    obj_catalog_ref = f"{current_obj_id} 0 R"
    pdf_content.append(f"{obj_catalog_ref.split(' ')[0]} 0 obj")
    pdf_content.append("<< /Type /Catalog")
    pdf_content.append(f"/Pages {obj_pages_ref} >>")
    pdf_content.append("endobj")
    current_obj_id += 1

    # Cross-reference table (xref) and Trailer
    xref_offset = len("\n".join(pdf_content)) + 1 # +1 for newline before xref
    
    xref_table = []
    current_offset = 0
    obj_offsets = []

    # Calculate offsets for each object
    for line in pdf_content:
        if line.endswith("obj"):
            obj_offsets.append(current_offset)
        current_offset += len(line.encode('latin1')) + 1 # +1 for newline
    
    xref_table.append("xref")
    xref_table.append(f"0 {current_obj_id}") # 0 for free object list, then total objects
    xref_table.append("0000000000 65535 f ") # Free object entry

    for i in range(len(obj_offsets)):
        xref_table.append(f"{obj_offsets[i]:010d} 00000 n ")

    pdf_content.extend(xref_table)

    pdf_content.append("trailer")
    pdf_content.append("<< /Size {} /Root {} >>".format(current_obj_id, obj_catalog_ref))
    pdf_content.append("startxref")
    pdf_content.append(str(xref_offset))
    pdf_content.append("%%EOF")
    
    with open(filename, "wb") as f: # Use 'wb' for binary write
        f.write("\n".join(pdf_content).encode('latin1'))

if __name__ == "__main__":
    generate_simple_pdf()
